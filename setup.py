from setuptools import setup, find_packages

setup(
    name='vq_compress',
    author='Asif Ahmed',
    description='Image compression with vqgan, autoencoder etc.',
    version='0.0.2',
    url='https://github.com/quickgrid/vq-compress',
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.11',
    install_requires=[
        'numpy',
        'omegaconf',
        'safetensors',
        'torch',
        'torchvision',
        'pillow',
        'tqdm'
    ]
)
