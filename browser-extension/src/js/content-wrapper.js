// This wrapper script handles loading the main content script as a module
(async () => {
    try {
        // Load config first
        const configScript = document.createElement('script');
        configScript.type = 'module';
        configScript.src = chrome.runtime.getURL('js/config.js');
        (document.head || document.documentElement).appendChild(configScript);

        // Wait for config to load before loading content script
        configScript.onload = () => {
            const contentScript = document.createElement('script');
            contentScript.type = 'module';
            contentScript.src = chrome.runtime.getURL('js/content.js');
            (document.head || document.documentElement).appendChild(contentScript);
        };
    } catch (error) {
        console.error('Error loading scripts:', error);
    }
})();