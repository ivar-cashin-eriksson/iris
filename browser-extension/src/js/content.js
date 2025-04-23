// Configuration
const config = {
  api: {
    baseUrl: 'http://localhost:5000',
    dbName: 'iris_dev_pas_normal_studios'
  },
  ui: {
    initialLoadDelay: 500,
    processingDebounceDelay: 100
  }
};

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

// UI Components
class UIComponents {
  static createOverlay(data) {
    const overlay = document.createElement('div');
    overlay.className = 'iris-overlay';
    
    const watermark = document.createElement('div');
    watermark.className = 'iris-watermark';

    if (!data.exists) {
      watermark.classList.add('iris-watermark-not-found');
      watermark.textContent = 'not found';
    } else if (!data.hasProductMasks) {
      watermark.classList.add('iris-watermark-found');
      watermark.textContent = 'iris';
    } else {
      watermark.classList.add('iris-watermark-found');
      watermark.textContent = 'iris - has_products';
    }
    
    overlay.appendChild(watermark);
    return overlay;
  }

  static createProductLink(mask, container) {
    const link = document.createElement('a');
    link.href = mask.product_url;
    link.className = 'iris-product-link';
    link.target = '_blank';

    const hotspot = this.createHotspot();
    const tooltip = this.createTooltip(mask);

    link.appendChild(hotspot);
    link.appendChild(tooltip);

    const containerRect = container.getBoundingClientRect();
    const { x, y } = mask.point;
    link.style.left = `${(x / containerRect.width) * 100}%`;
    link.style.top = `${(y / containerRect.height) * 100}%`;

    return link;
  }

  static createHotspot() {
    const hotspot = document.createElement('div');
    hotspot.className = 'iris-hotspot';

    const dotContainer = document.createElement('div');
    dotContainer.className = 'iris-hotspot-dot';

    const innerDot = document.createElement('div');
    innerDot.className = 'iris-hotspot-inner';

    dotContainer.appendChild(innerDot);
    hotspot.appendChild(dotContainer);
    return hotspot;
  }

  static createTooltip(mask) {
    const tooltip = document.createElement('div');
    tooltip.className = 'iris-tooltip';

    const tooltipContent = document.createElement('div');
    tooltipContent.className = 'iris-tooltip-content';

    const imageContainer = document.createElement('div');
    imageContainer.className = 'iris-tooltip-image-container';

    const productImage = document.createElement('img');
    productImage.className = 'iris-tooltip-image';
    productImage.src = mask.product_image || mask.product_url.replace('/products/', '/cdn/shop/products/') + '.jpg';
    productImage.alt = mask.product_title || '';
    productImage.loading = 'lazy';

    const infoContainer = document.createElement('div');
    infoContainer.className = 'iris-tooltip-info';

    if (mask.product_title) {
      const title = document.createElement('p');
      title.className = 'iris-tooltip-title';
      title.textContent = mask.product_title;
      infoContainer.appendChild(title);
    }

    if (mask.product_price) {
      const price = document.createElement('span');
      price.className = 'iris-tooltip-price';
      price.textContent = mask.product_price;
      infoContainer.appendChild(price);
    }

    const button = document.createElement('span');
    button.className = 'iris-tooltip-button';
    button.textContent = 'View product';

    imageContainer.appendChild(productImage);
    tooltipContent.appendChild(imageContainer);
    tooltipContent.appendChild(infoContainer);
    tooltipContent.appendChild(button);
    tooltip.appendChild(tooltipContent);

    return tooltip;
  }
}

// Product Overlay Manager
class ProductOverlayManager {
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
          mutation.addedNodes.forEach(node => {
            if (this.isImageNode(node)) {
              pendingImages.add(node);
              shouldProcess = true;
            }
          });

          mutation.removedNodes.forEach(node => {
            if (node.nodeType === Node.ELEMENT_NODE) {
              this.cleanupNode(node);
            }
          });
        }
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

// Initialize the product overlay manager
new ProductOverlayManager();
