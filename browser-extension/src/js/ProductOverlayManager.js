import { config } from './config.js';
import { DataManager } from './data/DataManager.js';
import { UIComponents } from './ui/UIComponents.js';

export class ProductOverlayManager {
  constructor() {
    this.dataManager = new DataManager();
    this.overlays = new Map();
    this.processingUrls = new Set();
    this.slideObservers = new WeakMap();
    
    this.setupMutationObserver();
    setTimeout(() => this.processExistingImages(), config.ui.initialLoadDelay);
  }

  setupMutationObserver() {
    let pendingImages = new Set();
    let processingTimeout = null;

    const observer = new MutationObserver((mutations) => {
      let shouldProcess = false;
      
      mutations.forEach(mutation => {
        if (mutation.type === 'childList') {
          // Handle added nodes
          mutation.addedNodes.forEach(node => {
            if (this.isImageNode(node)) {
              pendingImages.add(node);
              shouldProcess = true;
            }
          });

          // Clean up removed nodes
          mutation.removedNodes.forEach(node => {
            if (node.nodeType === Node.ELEMENT_NODE) {
              this.cleanupNode(node);
            }
          });
        }
        // Handle image src changes
        else if (this.isImageAttributeChange(mutation)) {
          pendingImages.add(mutation.target);
          shouldProcess = true;
        }
      });

      if (shouldProcess) {
        clearTimeout(processingTimeout);
        processingTimeout = setTimeout(() => {
          pendingImages.forEach(img => this.processImage(img));
          pendingImages.clear();
        }, config.ui.processingDebounceDelay);
      }
    });

    observer.observe(document.body, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ['src', 'srcset']
    });
  }

  isImageNode(node) {
    if (node.nodeName === 'IMG') return true;
    if (node.nodeType === Node.ELEMENT_NODE) {
      const images = node.querySelectorAll('img');
      images.forEach(img => this.processImage(img));
    }
    return false;
  }

  isImageAttributeChange(mutation) {
    return mutation.type === 'attributes' && 
           (mutation.attributeName === 'src' || mutation.attributeName === 'srcset') &&
           mutation.target.nodeName === 'IMG';
  }

  cleanupNode(node) {
    if (node.nodeName === 'IMG') {
      this.cleanupImage(node);
    }
    if (node.classList?.contains('swiper-slide')) {
      this.cleanupSlide(node);
    }
    node.querySelectorAll('img').forEach(img => this.cleanupImage(img));
  }

  cleanupImage(imgElement) {
    if (this.overlays.has(imgElement)) {
      this.overlays.get(imgElement).remove();
      this.overlays.delete(imgElement);
    }
  }

  cleanupSlide(slideElement) {
    if (this.slideObservers.has(slideElement)) {
      this.slideObservers.get(slideElement).disconnect();
      this.slideObservers.delete(slideElement);
    }
  }

  processExistingImages() {
    document.querySelectorAll('img').forEach(img => this.processImage(img));
  }

  async processImage(imgElement) {
    if (!this.isValidImage(imgElement)) return;
    if (this.processingUrls.has(imgElement.src)) return;
    
    try {
      this.processingUrls.add(imgElement.src);
      const data = await this.dataManager.checkImage(imgElement.src);
      this.createOrUpdateOverlay(imgElement, data);
    } catch (error) {
      console.error('Error processing image:', error);
      this.createOrUpdateOverlay(imgElement, { exists: false, error: error.message });
    } finally {
      this.processingUrls.delete(imgElement.src);
    }
  }

  isValidImage(imgElement) {
    return imgElement && imgElement.src && !imgElement.src.startsWith('data:');
  }

  createOrUpdateOverlay(imgElement, data) {
    const container = imgElement.closest('figure') || imgElement.parentElement;
    if (!container) return;

    this.cleanupImage(imgElement);

    const overlay = UIComponents.createOverlay(data);
    
    if (data.masks?.length) {
      data.masks.forEach(mask => {
        if (mask.point && mask.product_url) {
          const link = UIComponents.createProductLink(mask, container);
          overlay.appendChild(link);
        }
      });
    }

    this.ensureContainerPositioning(container);
    container.appendChild(overlay);
    this.overlays.set(imgElement, overlay);
    
    if (container.classList.contains('swiper-slide')) {
      this.setupSlideObserver(container, overlay);
    }
  }

  ensureContainerPositioning(container) {
    const containerStyle = window.getComputedStyle(container);
    if (containerStyle.position === 'static') {
      container.style.position = 'relative';
    }
  }

  setupSlideObserver(container, overlay) {
    this.cleanupSlide(container);

    const updateVisibility = () => {
      const isActive = container.classList.contains('swiper-slide-active') ||
                      container.classList.contains('swiper-slide-next') ||
                      container.classList.contains('swiper-slide-prev');
      overlay.style.display = isActive ? 'block' : 'none';
    };
    
    updateVisibility();

    const observer = new MutationObserver(updateVisibility);
    observer.observe(container, {
      attributes: true,
      attributeFilter: ['class']
    });
    this.slideObservers.set(container, observer);
  }
}