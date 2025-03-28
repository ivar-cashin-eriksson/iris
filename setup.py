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
        "opencv-python",
        "matplotlib",
        "numpy",
        "torch",
        "Pillow",
        "pycocotools",
        "tqdm",
    ],
    extras_require={
        "dev": [
            "isort>=5.12.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
    },
)