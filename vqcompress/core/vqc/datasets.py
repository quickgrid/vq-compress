import json
import os
from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from safetensors import safe_open
from torch.utils.data import Dataset
from torchvision import transforms

from vqcompress.core.ldm.util import preprocess_vqgan
from vqcompress.core.vqc.config import CompressionConfig


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
                    os.path.join(path.parent.absolute(), data[CompressionConfig.key_output_filename]),
                    framework="pt",
                    device="cuda"
                    # device="cpu"
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
        self.files_list = [
            p for ext in CompressionConfig.dataset_image_exts for p in Path(f'{input_path}').glob(f'*.{ext}')
        ]

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
