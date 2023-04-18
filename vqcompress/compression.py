import argparse
import json
import os
import pathlib
from typing import List

import torch
from omegaconf import OmegaConf
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from tqdm import tqdm

from vqcompress.core.ldm.util import custom_to_pil
from vqcompress.core.ldm.util import instantiate_from_config
from vqcompress.core.vqc.config import CompressionConfig
from vqcompress.core.vqc.datasets import CustomImageDataset, CustomEncodingDataset

torch.set_grad_enabled(False)


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
            use_float16_precision: bool = True,
            use_xformers: bool = False,
    ):
        self.use_xformers = use_xformers
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

        self.precision_type = torch.float16 if use_float16_precision else torch.float32

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
            pin_memory=True,
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
        sd_keys = sd.keys()

        def delete_model_layers(layer_initial_list: List[str]):
            for layer_initial in layer_initial_list:
                key_delete_list = []
                for dkey in sd_keys:
                    if dkey.split('.')[0] == layer_initial:
                        key_delete_list.append(dkey)

                for k in key_delete_list:
                    del sd[f'{k}']

        delete_model_layers(['loss', 'model_ema'])

        if use_decompress:
            delete_model_layers(['quant_conv', 'encoder'])
        else:
            delete_model_layers(['post_quant_conv', 'decoder'])

        if use_xformers:
            import vqcompress.core.vqc.code_patching
            import vqcompress.core.ldm.model
            vqcompress.core.ldm.model.AttnBlock.forward = vqcompress.core.vqc.code_patching.patch_xformers_attn_forward

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
                    with torch.autocast(device_type=self.device, dtype=self.precision_type):
                        xrec = self.ldm_model.decode(input_data)

                    for idx, fname in enumerate(full_filename_list):
                        x0 = custom_to_pil(xrec[idx])
                        x0.save(os.path.join(self.output_path, f"{fname[0]}"))
                else:
                    if self.vq_ind:
                        if 'ind_16' in loaded_json_data.keys():
                            input_data = input_data.type(torch.int64)

                        if 'z_shape' in loaded_json_data.keys():
                            z_shaped_loaded = loaded_json_data['z_shape']

                        out = self.ldm_model.quantize.get_codebook_entry(
                            input_data, tuple(z_shaped_loaded)
                        )

                        with torch.autocast(device_type=self.device, dtype=self.precision_type):
                            xrec = self.ldm_model.decode(out)

                        for idx, fname in enumerate(full_filename_list):
                            x0 = custom_to_pil(xrec[idx])
                            x0.save(os.path.join(self.output_path, f"{fname[0]}"))

                    else:
                        with torch.autocast(device_type=self.device, dtype=self.precision_type):
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
                    with torch.autocast(device_type=self.device, dtype=self.precision_type):
                        posterior = self.ldm_model.encode(input_data)
                        compressed_tensor = posterior.sample()

                else:
                    if self.vq_ind:
                        with torch.autocast(device_type=self.device, dtype=self.precision_type):
                            z, _, [_, _, indices] = self.ldm_model.encode(input_data)

                        compressed_tensor = indices
                        z_shape = (z.shape[0], z.shape[2], z.shape[3], z.shape[1])
                        json_data_obj['z_shape'] = z_shape

                        if self.use_16_bit_indices:
                            compressed_tensor = compressed_tensor.type(torch.int16)
                            json_data_obj["ind_16"] = 1

                    else:
                        with torch.autocast(device_type=self.device, dtype=self.precision_type):
                            z, _, [_, _, indices] = self.ldm_model.encode(input_data)

                        # input_img_size / downscale value for h, w. For vq-f4 downsample by 4.
                        z_shape = (z.shape[0], z.shape[2], z.shape[3], z.shape[1])
                        compressed_tensor = self.ldm_model.quantize.get_codebook_entry(
                            indices, z_shape
                        )

                tensors['weight'] = compressed_tensor
                output_file_name = f"batch_{batch_idx}.safetensors"

                save_file(tensors, os.path.join(self.output_path, output_file_name))

                json_data_obj[CompressionConfig.key_output_filename] = output_file_name
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
    parser.add_argument('--batch', help="process image count", default=1, type=int)
    parser.add_argument('--kl', help="generates output from kl autoencoder", action='store_true')
    parser.add_argument('--dc', help="decompresses encoding", action='store_true')
    parser.add_argument('--vq_ind', help="generates vqgan encoding instead of indices", action='store_true')
    parser.add_argument('--aspect', help="preserves aspect ratio resize by given image size", action='store_true')
    parser.add_argument('--ind_16', help="saves vq indices as 16 bit tensors", action='store_true')
    parser.add_argument('--float16', help="process in half precision", action='store_true')
    parser.add_argument('--xformers', help="use xformers to save memory and speedup in some cases", action='store_true')

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
        use_float16_precision=args.float16,
        use_xformers=args.xformers,
    )
    image_compression.process()


if __name__ == '__main__':
    main()
