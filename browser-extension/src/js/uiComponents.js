export class UIComponents {
    static createOverlay() {
        const overlay = document.createElement('div');
        overlay.className = 'image-overlay';
        return overlay;
    }

    static createProductLink(detection, container) {
        const link = document.createElement('a');
        link.className = 'iris-link';
        link.target = '_blank';
        
        // Set initial URL to first prediction
        link.href = detection.product_predictions[0].product_url;
        
        // Position the link using the normalized coordinates (0-1) from the API
        link.style.left = `${detection.point.x * 100}%`;
        link.style.top = `${detection.point.y * 100}%`;

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
        
        const tooltipContent = detection.product_predictions.map(product => `
            <div class="iris-tooltip-content" data-url="${product.product_url}">
                <img src="${product.product_image}" alt="${product.product_title}" class="iris-tooltip-image" />
                <div class="iris-tooltip-info">
                    <div class="iris-tooltip-title">${product.product_title}</div>
                    <div class="iris-tooltip-price">${product.product_price}</div>
                    ${product.score ? `<div class="iris-tooltip-score">Score: ${product.score.toFixed(3)}</div>` : ''}
                </div>
                <span class="iris-tooltip-button">View product</span>
            </div>
        `).join('');

        tooltip.innerHTML = `
            <div class="iris-tooltip-scroll">
                ${tooltipContent}
            </div>
        `;

        // Add click handlers for each product option
        tooltip.querySelectorAll('.iris-tooltip-content').forEach(content => {
            content.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                link.href = content.dataset.url;
                window.open(link.href, '_blank');
            });
        });

        link.appendChild(hotspot);
        link.appendChild(tooltip);
        
        return link;
    }
}
