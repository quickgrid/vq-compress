from dataclasses import dataclass


@dataclass
class Config:
    available_bitsandbytes: bool = False
    available_xformers: bool = False
