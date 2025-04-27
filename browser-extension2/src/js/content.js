class OverlayManager {
    constructor() {
        this.overlays = new Map();
        this.imageCache = {};
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
    processNode(node) {
        if (node.nodeName === 'IMG') {
            if (!this.imageCache[node.src]) {
                this.imageCache[node.src] = node.src;
            }
            this.createOverlay(node);
        } else if (node.nodeType === Node.ELEMENT_NODE) {
            node.querySelectorAll('img').forEach(img => {
                if (!this.imageCache[node.src]) {
                    this.imageCache[node.src] = node.src;
                }
                this.createOverlay(img)
            });
        }

    }

    cleanupNode(node) {
        if (node.nodeName === 'IMG') {
            this.removeOverlay(node);
        } else if (node.nodeType === Node.ELEMENT_NODE) {
            node.querySelectorAll('img').forEach(img => this.removeOverlay(img));
        }
    }

    processExistingImages() {
        document.querySelectorAll('img').forEach(img => this.createOverlay(img));
    }

    /** @param {HTMLElement} imgElement */
    createOverlay(imgElement) {
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


        // Add your custom overlay content here
        const content = document.createElement('div');
        content.className = 'overlay-content';
        overlay.appendChild(content);

        container.appendChild(overlay);
        this.overlays.set(imgElement, overlay);
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