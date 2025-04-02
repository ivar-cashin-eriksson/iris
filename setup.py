from setuptools import setup, find_packages

setup(
    name="iris",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "selenium",
        "beautifulsoup4",
        "pymongo",
        "webdriver-manager",
        "requests",
        "opencv-python>=4.7.0",
        "matplotlib>=3.7.0",
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "Pillow>=9.5.0",
        "pycocotools",
        "tqdm",
        "segment-anything>=1.0.0",
    ],
    extras_require={
        "dev": [
            "isort>=5.12.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
    },
)