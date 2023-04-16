# vq-compress

Image compression and reconstruction using pretrained autoencoder, vqgan first stage models from [latent diffusion](https://github.com/CompVis/latent-diffusion/tree/a506df5756472e2ebaf9078affdde2c4f1502cd4) and [taming transformer](https://github.com/CompVis/taming-transformers/tree/3ba01b241669f5ade541ce990f7650a3b8f65318) repo. Model codes, configs are copied from these repos with unnecessary parts removed.

To save vram and prevent extra processing only encoder or decoder weights based on compression or decompression task is loaded. Training code is removed but should be able to load models trained on original repo. 

Compressed data is saved in `safetensors` format. For compression if batch size larger than 1 is used then each output contains encode output tensor for the whole batch.

`vq-f4`, `vq-f8`, `kl-f4`, `kl-f8` configs provide the best reconstruction results.

Compressing with vqgan model by removing `--kl` and adding `--vq_ind` for `vq-f4`, `vq-f8` should provide best compression ratio. Further using a zip program to compress the saved output may provide better quality and file size reduction than using jpeg with quality reduced to around 60 percent. A good quality pretrained reconstruction model for `vq-f8` followed by zip compression may provide best results in terms of file size.



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


## Flags

If `--dc` flag is provided it runs decompression otherwise compresses input.

`--aspect` resize image keeping aspect ratio with smaller dimension size set to `--img_size`.

For `--ind_16` vqgan indices are saved as int16 reducing compressed output file size.

Currently 3 types of data compression is available. 
- For `--kl` autoencoder kl pretrained model encode output is saved.
- If `--kl` not specified then vqgan encode output is saved.
- If `--vq_ind` specified then indices are saved. These are used to reconstruct image.


## Pretrained Model and Configs

Original configs can be found [here](https://github.com/CompVis/latent-diffusion/tree/main/models/first_stage_models). More weights can be found on latent diffusion repo. [ru-dalle](https://github.com/ai-forever/ru-dalle/blob/1ab4e30ac14edd282e4abed57528eb97a9f2cb2e/rudalle/vae/__init__.py) `vq-f8-gumbel` model trained on taming transformers repo can also be used. 

For `kl-f8` stable diffusion vae ckpt can be used. Gives 8x downsampling.
- https://huggingface.co/stabilityai/sd-vae-ft-ema-original/tree/main
- https://huggingface.co/stabilityai/sd-vae-ft-mse-original/tree/main

For `kl-f4` config,
- https://ommer-lab.com/files/latent-diffusion/kl-f4.zip

For `vq-f4` config,
- https://ommer-lab.com/files/latent-diffusion/vq-f4.zip

Following may provide better compression rates but there maybe noticable degradation in reconstructed images.

For `vq-f8` config,
- https://ommer-lab.com/files/latent-diffusion/vq-f8.zip

For `vq-f8-n256` config,
- https://ommer-lab.com/files/latent-diffusion/vq-f8-n256.zip

For `kl-f16` config,
- https://ommer-lab.com/files/latent-diffusion/kl-f16.zip

For `kl-f32` config,
- https://ommer-lab.com/files/latent-diffusion/kl-f32.zip

For `vq-f8-gumbel` config, 
- https://heibox.uni-heidelberg.de/d/2e5662443a6b4307b470/

For `vq-f8-rudalle` config,
- https://huggingface.co/ai-forever/rudalle-utils/tree/main



## References

- https://github.com/CompVis/taming-transformers
- https://github.com/CompVis/latent-diffusion
