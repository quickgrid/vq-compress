from dataclasses import dataclass


@dataclass
class CompressionConfig:
    key_output_filename: str = "batched_file_name"
    dataset_image_exts = ['jpg', 'jpeg', 'png']
