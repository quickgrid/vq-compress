import argparse
import json
import os
import pathlib
from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from safetensors import safe_open
from safetensors.torch import save_file
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from vq_compress.core.ldm.util import instantiate_from_config

torch.set_grad_enabled(False)

exts = ['jpg', 'jpeg', 'png']


def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x


def preprocess_vqgan(x):
    return 2. * x - 1.


class CustomEncodingDataset(Dataset):
    def __init__(
            self,
            input_path: str,
            image_size: int,
    ) -> None:
        super().__init__()
        self.input_path = input_path
        self.image_size = image_size
        encoding_exts = ['json']
        self.files_list = [p for enc_ext in encoding_exts for p in Path(f'{input_path}').glob(f'*.{enc_ext}')]

    def __len__(self) -> int:
        return len(self.files_list)

    def __getitem__(self, index) -> Tuple[dict, dict]:
        path = self.files_list[index]

        with open(path, "r") as jsonfile:
            data = json.load(jsonfile)
            tensors = {}
            with safe_open(
                    os.path.join(path.parent.absolute(), data['batched_file_name']),
                    framework="pt",
                    device="cuda"
            ) as f:
                for key in f.keys():
                    tensors[key] = f.get_tensor(key)

        return tensors, data


class CustomImageDataset(Dataset):
    def __init__(
            self,
            input_path: str,
            image_size: int,
            keep_aspect_ratio: bool,
    ) -> None:
        super().__init__()

        self.input_path = input_path
        self.image_size = image_size
        self.files_list = [p for ext in exts for p in Path(f'{input_path}').glob(f'*.{ext}')]

        resize_type = transforms.Resize(image_size) if keep_aspect_ratio else transforms.Resize((image_size, image_size))

        self.transform = transforms.Compose([
            resize_type,
            transforms.ToTensor(),
            transforms.Lambda(preprocess_vqgan),
        ])

    def __len__(self) -> int:
        return len(self.files_list)

    def __getitem__(self, index) -> Tuple[torch.Tensor, str, str]:
        path = self.files_list[index]
        img = Image.open(path).convert('RGB')
        transformed_img = self.transform(img)
        return transformed_img, path.stem, path.name


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


class ImageCompression:
    def __init__(
            self,
            input_path: str,
            output_path: str,
            cfg_path: str,
            ckpt_path: str,
            image_size: int = 256,
            batch_size: int = 2,
            num_workers: int = 0,
            device: str = 'cuda',
            use_kl: bool = True,
            vq_ind: bool = True,
            use_decompress: bool = True,
            keep_aspect_ratio: bool = False,
            use_16_bit_indices: bool = True,
    ):
        self.use_16_bit_indices = use_16_bit_indices
        self.use_decompress = use_decompress
        self.vq_ind = vq_ind
        self.use_kl = use_kl
        self.device = device
        self.input_path = input_path
        self.output_path = output_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.output_path = self.output_path or self.input_path
        # dest_path = f'{args.dest}/output'
        pathlib.Path(input_path).mkdir(parents=True, exist_ok=True)
        pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

        img_dataset = CustomImageDataset(
            input_path=self.input_path,
            image_size=self.image_size,
            keep_aspect_ratio=keep_aspect_ratio,
        )
        self.img_data_loader = DataLoader(
            img_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            # pin_memory=True,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=collate_fn,
        )

        encoding_dataset = CustomEncodingDataset(
            input_path=self.input_path,
            image_size=self.image_size,
        )
        # For encoding to image processinge single item batch used.
        self.encoding_data_loader = DataLoader(
            encoding_dataset,
            batch_size=1,
            shuffle=False,
            # pin_memory=True,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=collate_fn,
        )

        # Only load weights for inference and delete unnecessary values to save vram. Works on kl-f8 config.
        config = OmegaConf.load(cfg_path)
        pl_sd = torch.load(ckpt_path, map_location="cpu")
        sd = pl_sd["state_dict"]

        key_delete_list = []
        for dkey in sd.keys():
            if dkey.split('.')[0] == 'loss':
                key_delete_list.append(dkey)

        for k in key_delete_list:
            del sd[f'{k}']

        key_delete_list = []
        for dkey in sd.keys():
            if dkey.split('.')[0] == 'model_ema':
                key_delete_list.append(dkey)

        for k in key_delete_list:
            del sd[f'{k}']

        key_delete_list = []
        if use_decompress:
            for dkey in sd.keys():
                if dkey.split('.')[0] == 'quant_conv':
                    key_delete_list.append(dkey)
            for dkey in sd.keys():
                if dkey.split('.')[0] == 'encoder':
                    key_delete_list.append(dkey)
        else:
            for dkey in sd.keys():
                if dkey.split('.')[0] == 'post_quant_conv':
                    key_delete_list.append(dkey)
            for dkey in sd.keys():
                if dkey.split('.')[0] == 'decoder':
                    key_delete_list.append(dkey)

        for k in key_delete_list:
            del sd[f'{k}']

        # print(sd.keys())
        self.ldm_model = instantiate_from_config(config.model)
        self.ldm_model.load_state_dict(sd, strict=False)
        self.ldm_model = self.ldm_model.to(device)
        self.ldm_model.eval()
        del sd
        del pl_sd

    def process(self) -> None:
        if self.use_decompress:
            self.decompress()
        else:
            self.compress()

    def decompress(self) -> None:
        with tqdm(self.encoding_data_loader) as pbar:
            for batch_idx, (input_data_dict, loaded_json_data) in enumerate(pbar):
                input_data = input_data_dict['weight']
                input_data = input_data.squeeze(0)
                input_data = input_data.to(self.device)

                full_filename_list = loaded_json_data['file_name_list']

                if self.use_kl:
                    xrec = self.ldm_model.decode(input_data)
                    for idx, fname in enumerate(full_filename_list):
                        x0 = custom_to_pil(xrec[idx])
                        x0.save(os.path.join(self.output_path, f"{fname[0]}"))
                else:
                    if self.vq_ind:
                        if 'ind_16' in loaded_json_data.keys():
                            input_data = input_data.type(torch.int64)

                        if 'z_shape' in input_data_dict.keys():
                            z_shaped_loaded = input_data_dict['z_shape']

                        z_shaped_loaded = z_shaped_loaded.cpu().numpy()[0]
                        out = self.ldm_model.quantize.get_codebook_entry(
                            input_data, tuple(z_shaped_loaded)
                        )

                        xrec = self.ldm_model.decode(out)
                        for idx, fname in enumerate(full_filename_list):
                            x0 = custom_to_pil(xrec[idx])
                            x0.save(os.path.join(self.output_path, f"{fname[0]}"))

                    else:
                        xrec = self.ldm_model.decode(input_data)
                        for idx, fname in enumerate(full_filename_list):
                            x0 = custom_to_pil(xrec[idx])
                            x0.save(os.path.join(self.output_path, f"{fname[0]}"))

    def compress(self) -> None:
        with tqdm(self.img_data_loader) as pbar:
            for batch_idx, (input_data, filename, full_filename) in enumerate(pbar):
                input_data = input_data.to(self.device)

                tensors = {}
                json_data_obj = {}

                if self.use_kl:
                    posterior = self.ldm_model.encode(input_data)
                    compressed_tensor = posterior.sample()
                else:
                    if self.vq_ind:
                        z, _, [_, _, indices] = self.ldm_model.encode(input_data)
                        compressed_tensor = indices
                        z_shape = (z.shape[0], z.shape[2], z.shape[3], z.shape[1])
                        z_shape = torch.tensor(list(z_shape), dtype=torch.int16)
                        tensors['z_shape'] = z_shape

                        if self.use_16_bit_indices:
                            compressed_tensor = compressed_tensor.type(torch.int16)
                            json_data_obj["ind_16"] = 1

                    else:
                        z, _, [_, _, indices] = self.ldm_model.encode(input_data)
                        # input_img_size / downscale value for h, w. For vq-f4 downsample by 4.
                        z_shape = (z.shape[0], z.shape[2], z.shape[3], z.shape[1])
                        compressed_tensor = self.ldm_model.quantize.get_codebook_entry(
                            indices, z_shape
                        )

                tensors['weight'] = compressed_tensor
                save_file(tensors, os.path.join(self.output_path, f"batch_{batch_idx}.safetensors"))

                json_data_obj["batched_file_name"] = f"batch_{batch_idx}.safetensors"
                json_data_obj["file_name_list"] = list(full_filename)
                json_object = json.dumps(json_data_obj, indent=4)

                with open(os.path.join(self.output_path, f"batch_{batch_idx}.json"), "w") as jsonfile:
                    jsonfile.write(json_object)


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate captions from images.')
    parser.add_argument('-s', '--src', required=True, help="path to source folder", type=pathlib.Path)
    parser.add_argument('-d', '--dest', help="path to destination folder", type=pathlib.Path)
    parser.add_argument('--device', help="use cpu or cuda gpu", default='cuda', type=str)
    parser.add_argument('--cfg', help="model config file path", type=pathlib.Path)
    parser.add_argument('--ckpt', help="pretrained encode decode model path for config file", type=pathlib.Path)
    parser.add_argument('--img_size', metavar='--img-size', help="image size for processing", default=512, type=int)
    parser.add_argument('--batch', help="process image count", default=2, type=int)
    parser.add_argument('--kl', help="generates output from kl autoencoder", action='store_true')
    parser.add_argument('--dc', help="decompresses encoding", action='store_true')
    parser.add_argument('--vq_ind', help="generates vqgan encoding instead of indices", action='store_true')
    parser.add_argument('--aspect', help="preserves aspect ratio resize by given image size", action='store_true')
    parser.add_argument('--ind_16', help="saves vq indices as 16 bit tensors", action='store_true')

    args = parser.parse_args()

    image_compression = ImageCompression(
        input_path=args.src,
        output_path=args.dest,
        cfg_path=args.cfg,
        ckpt_path=args.ckpt,
        image_size=args.img_size,
        use_kl=args.kl,
        vq_ind=args.vq_ind,
        use_decompress=args.dc,
        batch_size=args.batch,
        keep_aspect_ratio=args.aspect,
        use_16_bit_indices=args.ind_16,
    )
    image_compression.process()


if __name__ == '__main__':
    main()
