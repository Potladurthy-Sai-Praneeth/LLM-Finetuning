from setuptools import setup, find_packages
from pathlib import Path

# Read the long description from README
this_directory = Path(__file__).parent


setup(
    name="llm-finetuning-qlora",
    version="1.0.0",
    author="Potladurthy Sai Praneeth",
    description="Fine-tuning Large Vision-Language Models with QLoRA, FSDP, and DeepSpeed",
    # Package discovery
    packages=find_packages(exclude=["tests", "tests.*", "notebooks", "*.ipynb"]),
    py_modules=[
        "data_preprocessing",
        "finetuning",
        "inference",
        "parallel_finetune",
        "deepspeed_finetune",
        "distributed_training",
        "distributed_deepspeed",
    ],
    
    # Python version requirement
    python_requires=">3.8,<=3.12",
    
    # Dependencies
    install_requires=[
        "transformers",
        "peft",
        "accelerate",
        "trl",
        "bitsandbytes",
        "datasets",
        "torch",
        "torchvision",
        "torchaudio",
        "pillow",
        "PyYAML",
        "deepspeed",
    ],
        
)
