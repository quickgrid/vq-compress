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


def reconstruct_with_vqgan(x, model):
    # could also use model(x) for reconstruction but use explicit encoding and decoding here
    z, _, [_, _, indices] = model.encode(x)
    print(f"VQGAN --- {model.__class__.__name__}: latent shape: {z.shape[2:]}")
    print(x.shape, z.shape, indices.shape)
    print(indices)
    xrec = model.decode(z)
    return xrec


def reconstruct_with_vqgan_code(x, model):
    # could also use model(x) for reconstruction but use explicit encoding and decoding here
    z, _, [_, _, indices] = model.encode(x)
    print(f"VQGAN --- {model.__class__.__name__}: latent shape: {z.shape[2:]}")
    print(x.shape, z.shape, indices.shape)
    print(indices)

    # input_img_size / downscale value for h, w. For vq-f4 downsample by 4.
    out = model.quantize.get_codebook_entry(indices, (z.shape[0], z.shape[2], z.shape[3], z.shape[1]))
    print(torch.allclose(z, out))

    xrec = model.decode(out)
    return xrec


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

    def __getitem__(self, index) -> Tuple[torch.Tensor, List[str]]:
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

        return tensors['weight'], data['file_name_list']


class CustomImageDataset(Dataset):
    def __init__(
            self,
            input_path: str,
            image_size: int,
    ) -> None:
        super().__init__()

        self.input_path = input_path
        self.image_size = image_size
        self.files_list = [p for ext in exts for p in Path(f'{input_path}').glob(f'*.{ext}')]

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
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
    ):
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
        del sd['model_ema.decay']
        del sd['model_ema.num_updates']

        key_delete_list = []

        if use_decompress:
            del sd['quant_conv.weight']
            del sd['quant_conv.bias']
            for dkey in sd.keys():
                if dkey.split('.')[0] == 'encoder':
                    key_delete_list.append(dkey)
        else:
            del sd['post_quant_conv.weight']
            del sd['post_quant_conv.bias']
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
            for batch_idx, (input_data, full_filename_list) in enumerate(pbar):
                input_data = input_data.squeeze(0)
                input_data = input_data.to(self.device)

                if self.use_kl:
                    xrec = self.ldm_model.decode(input_data)
                    for idx, fname in enumerate(full_filename_list):
                        x0 = custom_to_pil(xrec[idx])
                        x0.save(os.path.join(self.output_path, f"{fname[0]}"))

    def compress(self) -> None:
        with tqdm(self.img_data_loader) as pbar:
            for batch_idx, (input_data, filename, full_filename) in enumerate(pbar):
                input_data = input_data.to(self.device)

                if self.use_kl:
                    print(input_data.shape)
                    posterior = self.ldm_model.encode(input_data)
                    posterior_sample = posterior.sample()

                    tensors = {
                        "weight": posterior_sample,
                    }
                    save_file(tensors, os.path.join(self.output_path, f"batch_{batch_idx}.safetensors"))

                    json_filenames = {
                        "batched_file_name": f"batch_{batch_idx}.safetensors",
                        "file_name_list": list(full_filename),
                    }
                    json_object = json.dumps(json_filenames, indent=4)

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
    )
    image_compression.process()


if __name__ == '__main__':
    main()
