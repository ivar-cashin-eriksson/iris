import { config } from './config.js';
import { UIComponents } from './uiComponents.js';

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
      const url = new URL(`${config.api.baseUrl}/get-detections-all-predictions`);
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
        this.processQueue = [];
        this.isProcessing = false;
        this.mutationTimeout = null;
        this.setupMutationObserver();
        this.processExistingImages();
    }

    setupMutationObserver() {
        const observer = new MutationObserver((mutations) => {
            // Debounce mutation callbacks
            if (this.mutationTimeout) {
                clearTimeout(this.mutationTimeout);
            }
            this.mutationTimeout = setTimeout(() => {
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
            }, 500); // Wait 500ms before processing mutations
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
                this.addToQueue(node);
            }
        } else if (node.nodeType === Node.ELEMENT_NODE) {
            const images = node.querySelectorAll('img');
            for (const img of images) {
                if (!this.processingUrls.has(img.src)) {
                    this.addToQueue(img);
                }
            }
        }
    }

    addToQueue(imgElement) {
        this.processQueue.push(imgElement);
        this.processNextInQueue();
    }

    async processNextInQueue() {
        if (this.isProcessing || this.processQueue.length === 0) return;
        
        this.isProcessing = true;
        const imgElement = this.processQueue.shift();
        
        try {
            await this.processImage(imgElement);
        } catch (error) {
            console.error('Error processing image:', error);
        } finally {
            this.isProcessing = false;
            // Process next item after a small delay
            setTimeout(() => this.processNextInQueue(), 100);
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
        if (!imgElement || !imgElement.src || imgElement.src.startsWith('data:')) return false;
        
        const rect = imgElement.getBoundingClientRect();
        return Math.min(rect.width, rect.height) >= 256;
    }

    cleanupNode(node) {
        if (node.nodeName === 'IMG') {
            this.removeOverlay(node);
        } else if (node.nodeType === Node.ELEMENT_NODE) {
            node.querySelectorAll('img').forEach(img => this.removeOverlay(img));
        }
    }

    processExistingImages() {
        // Process visible images first
        const images = Array.from(document.querySelectorAll('img'));
        const visibleImages = images.filter(img => this.isElementInViewport(img));
        
        visibleImages.forEach(img => this.addToQueue(img));
        
        // Process remaining images
        images
            .filter(img => !visibleImages.includes(img))
            .forEach(img => this.addToQueue(img));
    }

    isElementInViewport(el) {
        const rect = el.getBoundingClientRect();
        return (
            rect.top >= 0 &&
            rect.left >= 0 &&
            rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
            rect.right <= (window.innerWidth || document.documentElement.clientWidth)
        );
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

        const overlay = UIComponents.createOverlay();
        
        const imageOffsetX = imgElement.getBoundingClientRect().x - container.getBoundingClientRect().x;
        overlay.style.left = `${imageOffsetX}px`;
        const imageOffsetY = imgElement.getBoundingClientRect().y - container.getBoundingClientRect().y;
        overlay.style.top = `${imageOffsetY}px`;
        overlay.style.width = `${imgElement.scrollWidth}px`;
        overlay.style.height = `${imgElement.scrollHeight}px`;

        if (data.detections.length === 0) {
            overlay.classList.add('no-products');
        }
        
        data.detections.forEach(detection => {
            const productLink = UIComponents.createProductLink(detection, container);
            overlay.appendChild(productLink);
        });

        this.overlays.set(imgElement, overlay);
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