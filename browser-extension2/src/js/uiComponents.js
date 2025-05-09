export class UIComponents {
    static createOverlay() {
        const overlay = document.createElement('div');
        overlay.className = 'image-overlay';
        return overlay;
    }

    static createProductLink(localization, container) {
        const link = document.createElement('a');
        link.href = localization.product_url;
        link.className = 'iris-link';
        link.target = '_blank';
        
        // Position the link using the normalized coordinates (0-1) from the API
        link.style.left = `${localization.point.x * 100}%`;
        link.style.top = `${localization.point.y * 100}%`;

        // Make sure the link takes the click event
        link.addEventListener('click', e => {
            e.stopPropagation();
        });

        const hotspot = document.createElement('div');
        hotspot.className = 'iris-hotspot';

        const dotContainer = document.createElement('div');
        dotContainer.className = 'iris-hotspot-dot';

        const innerDot = document.createElement('div');
        innerDot.className = 'iris-hotspot-inner';

        dotContainer.appendChild(innerDot);
        hotspot.appendChild(dotContainer);

        const tooltip = document.createElement('div');
        tooltip.className = 'iris-tooltip';
        
        tooltip.innerHTML = `
            <div class="iris-tooltip-content">
                <img src="${localization.product_image}" alt="${localization.product_title}" class="iris-tooltip-image" />
                <div class="iris-tooltip-info">
                    <div class="iris-tooltip-title">${localization.product_title}</div>
                    <div class="iris-tooltip-price">${localization.product_price}</div>
                </div>
                <span class="iris-tooltip-button">View product</span>
            </div>
        `;

        link.appendChild(hotspot);
        link.appendChild(tooltip);
        
        return link;
    }
}
