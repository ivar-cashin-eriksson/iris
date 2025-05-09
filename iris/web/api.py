from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from urllib.parse import unquote
import logging
from pymongo import MongoClient

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MongoDB client with Atlas connection string
client = MongoClient("mongodb+srv://test:test@iris-cluster.andes.mongodb.net/?retryWrites=true&w=majority&appName=iris-cluster")

@app.get("/check-url")
async def check_url(url: str = Query(...), db_name: str = Query(...)):
    """
    Check if a URL exists in the image_metadata collection and return localization information.
    """
    try:
        db = client[db_name]
        image_doc = db.image_metadata.find_one({"original_url": url})
        
        if not image_doc:
            return {
                "exists": False,
                "has_product_localizations": False,
                "localizations": []
            }

        # Check if any localizations have product_hash
        has_product_localizations = False
        if "localizations" in image_doc and image_doc["localizations"]:
            has_product_localizations = any(
                "product_hash" in localization 
                for localization in image_doc["localizations"]
            )

        response = {
            "exists": True,
            "has_product_localizations": has_product_localizations,
            "localizations": []
        }
        
        if "localizations" in image_doc and image_doc["localizations"]:
            for localization in image_doc["localizations"]:
                if "localization_point" in localization and "product_hash" in localization:
                    # Look up product information
                    product = db.products.find_one({"product_hash": localization["product_hash"]})
                    if product:
                        # Get first image URL from the product's image_hashes
                        product_image_url = None
                        if product.get("image_hashes") and len(product["image_hashes"]) > 0:
                            first_image_hash = product["image_hashes"][0]
                            image_meta = db.image_metadata.find_one({"image_hash": first_image_hash})
                            if image_meta:
                                product_image_url = image_meta.get("original_url")

                        response["localizations"].append({
                            "point": localization["localization_point"],
                            "product_url": product.get("url"),
                            "product_title": product.get("title"),
                            "product_price": product.get("price"),
                            "product_image": product_image_url
                        })
        
        logger.info(f"Checked URL {url} in database {db_name}: exists={response['exists']}, has_product_localizations={response['has_product_localizations']}")
        return response
    
    except Exception as e:
        logger.error(f"Error checking URL: {str(e)}", exc_info=True)
        return {"error": str(e), "exists": False, "has_product_localizations": False, "localizations": []}

@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=5000)