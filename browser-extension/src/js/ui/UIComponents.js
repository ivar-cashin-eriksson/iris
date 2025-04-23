export class UIComponents {
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