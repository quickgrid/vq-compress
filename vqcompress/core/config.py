from dataclasses import dataclass


@dataclass
class CompressionConfig:
    output_filename_key: str = "batched_file_name"
    dataset_image_exts = ['jpg', 'jpeg', 'png']
    saved_indices_precision_key = "saved_indices_precision"
