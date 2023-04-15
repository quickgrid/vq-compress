# vq-compress

Image compression and reconstruction using pretrained autoencoder, vqgan first stage models from [latent diffusion](https://github.com/CompVis/latent-diffusion/tree/a506df5756472e2ebaf9078affdde2c4f1502cd4) and [taming transformer](https://github.com/CompVis/taming-transformers/tree/3ba01b241669f5ade541ce990f7650a3b8f65318) repo. Model codes, configs are copied from these repos with unnecessary parts removed.

To save vram and prevent extra processing only encoder or decoder weights based on compression or decompression task is loaded. Training code is removed but should be able to load models trained on original repo. 

Compressed data is saved in `safetensors` format. For compression if batch size larger than 1 is used then each output contains encode output tensor for the whole batch.

Only `vq-f4`, `vq-f8`, `kl-f4`, `kl-f8` configs tested as these provide best reconstruction results.

Compressing with vqgan model by removing `--kl` and adding `--vq_ind` for `vq-f4`, `vq-f8` should provide best compression ratio. Further using a zip program to compress the saved output may provide better quality and file size reduction than using jpeg with quality reduced to around 60 percent. A good quality pretrained reconstruction model for `vq-f8` followed by zip compression may provide best results in terms of file size.


## Flags

If `--dc` flag is provided it runs decompression otherwise compresses input.

`--aspect` resize image keeping aspect ratio with smaller dimension size set to `--img_size`.

Currently 3 types of data compression is available. 
- For `--kl` autoencoder kl pretrained model encode output is saved.
- If `--kl` not specified then vqgan encode output is saved.
- If `--vq_ind` specified then indices are saved. These are used to reconstruct image.


## Pretrained Model and Configs

Original configs can be found [here](https://github.com/CompVis/latent-diffusion/tree/main/models/first_stage_models). More weights can be found on latent diffusion repo.

For `kl-f8` stable diffusion vae ckpt can be used. Gives 8x downsampling.
- https://huggingface.co/stabilityai/sd-vae-ft-ema-original/tree/main
- https://huggingface.co/stabilityai/sd-vae-ft-mse-original/tree/main

For `kl-f4` model,
- https://ommer-lab.com/files/latent-diffusion/kl-f4.zip

For `vq-f4` model,
- https://ommer-lab.com/files/latent-diffusion/vq-f4.zip


## Install 

Run following command on setup.py folder before running library.
> pip install -e .



## Commands

### Compress

kl compress,

> python compression.py -s "SRC_PATH" -d "DEST_PATH" --cfg "CONFIG_YAML_PATH" --ckpt "VAE_CKPT_PATH" --kl --batch 2 --img_size 384

vq compress with indices,

> python compression.py -s "SRC_PATH" -d "DEST_PATH" --cfg "CONFIG_YAML_PATH" --ckpt "VAE_CKPT_PATH" --batch 1 --img_size 512 --vq_ind


### Decompress

kl decompress,

> python compression.py -s "SRC_PATH" -d "DEST_PATH" --cfg "CONFIG_YAML_PATH" --ckpt "VAE_CKPT_PATH" --kl --dc

vq decompress with indices,

> python compression.py -s "SRC_PATH" -d "DEST_PATH" --cfg "CONFIG_YAML_PATH" --ckpt "VAE_CKPT_PATH" --dc --vq_ind


## References

- https://github.com/CompVis/taming-transformers
- https://github.com/CompVis/latent-diffusion
