// This wrapper script handles loading the main content script as a module
(async () => {
  try {
    // Create a script element to load the module
    const script = document.createElement('script');
    script.type = 'module';
    script.src = chrome.runtime.getURL('js/content.js');
    
    // Append it to either the head or body
    (document.head || document.documentElement).appendChild(script);
  } catch (error) {
    console.error('Error loading content script:', error);
  }
})();