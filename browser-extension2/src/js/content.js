import { config } from './config.js';

// Data Management
class DataManager {
  constructor() {
    this.cache = new Map();
  }

  async checkImage(imageUrl) {
    if (this.cache.has(imageUrl)) {
      return this.cache.get(imageUrl);
    }

    try {
      const url = new URL(`${config.api.baseUrl}/check-url`);
      url.searchParams.append('url', imageUrl);
      url.searchParams.append('db_name', config.api.dbName);

      const response = await fetch(url.toString(), {
        method: 'GET',
        headers: { 'Accept': 'application/json' }
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      this.cache.set(imageUrl, data);
      return data;
    } catch (error) {
      console.error('Error checking image:', error);
      throw error;
    }
  }

  clearCache() {
    this.cache.clear();
  }
}

class OverlayManager {
    constructor() {
        this.overlays = new Map();
        this.dataManager = new DataManager();
        this.processingUrls = new Set();
        this.setupMutationObserver();
        this.processExistingImages();
    }

    setupMutationObserver() {
        const observer = new MutationObserver((mutations) => {
            mutations.forEach(mutation => {
                if (mutation.type === 'childList') {
                    mutation.addedNodes.forEach(node => this.processNode(node));
                    mutation.removedNodes.forEach(node => this.cleanupNode(node));
                }
                if (mutation.type === 'attributes') {
                    this.cleanupNode(mutation.target);
                    this.processNode(mutation.target);
                }
            });
        });

        observer.observe(document.body, {
            childList: true,
            subtree: true,
            attributes: true,
            attributeFilter: ['src', 'srcset']
        });
    }

    /** @param {HTMLElement} imgElement */
    async processNode(node) {
        if (node.nodeName === 'IMG') {
            if (!this.processingUrls.has(node.src)) {
                await this.processImage(node);
            }
        } else if (node.nodeType === Node.ELEMENT_NODE) {
            const images = node.querySelectorAll('img');
            for (const img of images) {
                if (!this.processingUrls.has(img.src)) {
                    await this.processImage(img);
                }
            }
        }
    }

    async processImage(imgElement) {
        if (!this.isValidImage(imgElement)) return;
        
        try {
            this.processingUrls.add(imgElement.src);
            const data = await this.dataManager.checkImage(imgElement.src);
            this.createOverlay(imgElement, data);
        } catch (error) {
            console.error('Error processing image:', error);
            this.createOverlay(imgElement, { exists: false, error: error.message });
        } finally {
            this.processingUrls.delete(imgElement.src);
        }
    }

    isValidImage(imgElement) {
        return imgElement && imgElement.src && !imgElement.src.startsWith('data:');
    }

    cleanupNode(node) {
        if (node.nodeName === 'IMG') {
            this.removeOverlay(node);
        } else if (node.nodeType === Node.ELEMENT_NODE) {
            node.querySelectorAll('img').forEach(img => this.removeOverlay(img));
        }
    }

    processExistingImages() {
        document.querySelectorAll('img').forEach(img => this.processImage(img));
    }

    /**
     *  @param {HTMLElement} imgElement
     *  @param {Object} data
    */
    createOverlay(imgElement, data) {
        if (this.overlays.has(imgElement)) return;

        const container = getFirstParentWithBox(imgElement);
        if (!container) return;

        // Ensure container is positioned
        const containerStyle = window.getComputedStyle(container);
        if (containerStyle.position === 'static') {
            container.style.position = 'relative';
        }

        const overlay = document.createElement('div');
        overlay.className = 'image-overlay';
        
        const imageOffsetX = imgElement.getBoundingClientRect().x - container.getBoundingClientRect().x;
        overlay.style.left = `${imageOffsetX}px`;
        const imageOffsetY = imgElement.getBoundingClientRect().y - container.getBoundingClientRect().y;
        overlay.style.top = `${imageOffsetY}px`;
        overlay.style.width = `${imgElement.scrollWidth}px`;
        overlay.style.height = `${imgElement.scrollHeight}px`;

        if (data.masks.length===0) {
            overlay.classList.add('no-products');
        }
        this.overlays.set(imgElement, overlay);

        data.masks.forEach(mask => {
            const maskElement = document.createElement('a');
            maskElement.href = mask.product_url;
            maskElement.className = 'iris-link';
            maskElement.style.left = mask.point.x / imgElement.naturalWidth * 100 + '%';
            maskElement.style.top = mask.point.y / imgElement.naturalHeight * 100 + '%';
            maskElement.target = '_blank';
            maskElement.innerHTML = `
            <div class="iris-circle"></div>
            <div class="iris-product-container">
                <img src="${mask.product_image}" alt="${mask.product_title}" class="iris-product-image">
                <div class="iris-product-info">
                    <div>${mask.product_title}</div>
                    <div>${mask.product_price}</div>
                </div>
            </div>
            `;

            overlay.appendChild(maskElement);
        });

        container.appendChild(overlay);
    }


    removeOverlay(imgElement) {
        const overlay = this.overlays.get(imgElement);
        if (overlay) {
            overlay.remove();
            this.overlays.delete(imgElement);
        }
    }
}

// Initialize the overlay manager
new OverlayManager();

function getFirstParentWithBox(element) {
    let parent = element.parentElement;
    while (parent && getComputedStyle(parent).display === 'contents') {
        parent = parent.parentElement;
    }
    return parent;
}