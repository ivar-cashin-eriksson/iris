# iris 👁️🌈🧠

An intelligent product recognition system that detects and identifies products in e-commerce images using advanced computer vision. The system uses SAM2 or YOLOS for precise localization and CLIP for semantic understanding of products.

## 🌟 Features

- 🔍 Automatic product detection in images using SAM
- 🎯 Smart product linking using CLIP embeddings
- 🌐 Chrome extension for real-time product link display from pre-detected products
- 💾 MongoDB-based data persistence
- 🖼️ Multi-shop support
- 🚀 FastAPI backend

## 🏗️ Project Structure

```bash
iris/
├── browser-extension/           # Chrome extension
│
├── iris/                        # Core Python package
│   ├── config/                  # Configuration management
│   ├── data_pipeline/           # Data collection & MongoDB
│   ├── embedding_pipeline/      # CLIP embeddings
│   ├── localization_pipeline/   # Object localization
│   ├── utils/                   # Helper functions
│   └── web/                     # FastAPI server
│
└── test/                        # Test notebooks
```

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Node.js 16+ (for extension)
- MongoDB
- CUDA-capable GPU (recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/iris.git
cd iris
```

2. Install Python dependencies:
```bash
pip install -e .
```

3. Download SAM checkpoint:
```bash
mkdir -p checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P checkpoints/
```

4. Set up shop configs:
   - Copy example shop config from `configs/data_pipeline/shops/`
   - Update with your shop's details

5. Set up the extension:
```bash
cd browser-extension
npm install
```

## 🎮 Development

### API Server

Start in development mode with hot-reload:
```bash
uvicorn iris.web.api:app --reload --host 0.0.0.0 --port 5000
```

### Chrome Extension

1. Open Chrome and go to `chrome://extensions/`
2. Enable "Developer mode"
3. Click "Load unpacked" and select `browser-extension/src`

### Testing

Explore the notebooks in `test/`:
- `product_linker_test.ipynb`: Test product linking
- `localization_test.ipynb`: Test object localization
- `embedding_test.ipynb`: Test CLIP embeddings
- `data_utils_test.ipynb`: Test data utilities

## 💡 Core Components

### Object Localization Pipeline
Uses Meta's Segment Anything Model or YOLOS to detect and localise products in images with high precision.

### Embedding Pipeline
Leverages OpenAI's CLIP model to understand product semantics and create meaningful embeddings for similarity matching.

### Data Pipeline
Manages data collection, storage, and MongoDB interactions. Supports multiple e-commerce platforms through a flexible configuration system.

### Browser Extension
Adds interactive product links to e-commerce images in real-time by matching them against a database of previously detected and analyzed products. Built with modern JavaScript and supports dynamic page updates.

## 📝 License

This software is proprietary and confidential. All rights reserved. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Segment Anything Model (SAM)](https://segment-anything.com/) by Meta AI Research
- [CLIP](https://openai.com/blog/clip/) by OpenAI