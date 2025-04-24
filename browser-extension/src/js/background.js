import { config } from './config.js';

// Cache for API responses
const imageCache = new Map();

// Listen for messages from the content script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.type === 'CHECK_IMAGE') {
    // Check cache first
    if (imageCache.has(request.imageUrl)) {
      const cachedData = imageCache.get(request.imageUrl);
      sendResponse(cachedData);
      return true;
    }

    // If not in cache, make API request
    checkImage(request.imageUrl)
      .then(data => {
        // Cache the response
        imageCache.set(request.imageUrl, data);
        sendResponse(data);
      })
      .catch(error => {
        console.error('Error in background script:', error);
        const errorResponse = { 
          error: error.message, 
          exists: false,
          hasProductMasks: false,
          masks: [] 
        };
        imageCache.set(request.imageUrl, errorResponse);
        sendResponse(errorResponse);
      });
    
    return true; // Keep the message channel open for async response
  }
});

async function checkImage(imageUrl) {
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
    return {
      exists: data.exists,
      hasProductMasks: data.has_product_masks,
      masks: data.masks || [],
      error: null
    };
  } catch (error) {
    throw error;
  }
}