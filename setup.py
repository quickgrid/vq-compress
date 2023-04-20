from setuptools import setup, find_packages

setup(
    name='vqcompress',
    author='Asif Ahmed',
    description='Image compression with vqgan, autoencoder etc.',
    version='0.2.0',
    url='https://github.com/quickgrid/vq-compress',
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    install_requires=[
        'numpy',
        'omegaconf',
        'safetensors',
        'torch',
        'torchvision',
        'pillow',
        'tqdm',
        'einops',
        'lightning',
        'xformers'
    ]
)
