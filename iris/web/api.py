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
    Check if a URL exists in the image_metadata collection and return mask information.
    """
    try:
        db = client[db_name]
        image_doc = db.image_metadata.find_one({"original_url": url})
        
        if not image_doc:
            return {
                "exists": False,
                "has_product_masks": False,
                "masks": []
            }

        # Check if any masks have product_hash
        has_product_masks = False
        if "masks" in image_doc and image_doc["masks"]:
            has_product_masks = any(
                "product_hash" in mask 
                for mask in image_doc["masks"]
            )

        response = {
            "exists": True,
            "has_product_masks": has_product_masks,
            "masks": []
        }
        
        if "masks" in image_doc and image_doc["masks"]:
            for mask in image_doc["masks"]:
                if "mask_point" in mask and "product_hash" in mask:
                    # Look up product information
                    product = db.products.find_one({"product_hash": mask["product_hash"]})
                    if product:
                        # Get first image URL from the product's image_hashes
                        product_image_url = None
                        if product.get("image_hashes") and len(product["image_hashes"]) > 0:
                            first_image_hash = product["image_hashes"][0]
                            image_meta = db.image_metadata.find_one({"image_hash": first_image_hash})
                            if image_meta:
                                product_image_url = image_meta.get("original_url")

                        response["masks"].append({
                            "point": mask["mask_point"],
                            "product_url": product.get("url"),
                            "product_title": product.get("title"),
                            "product_price": product.get("price"),
                            "product_image": product_image_url
                        })
        
        logger.info(f"Checked URL {url} in database {db_name}: exists={response['exists']}, has_product_masks={response['has_product_masks']}")
        return response
    
    except Exception as e:
        logger.error(f"Error checking URL: {str(e)}", exc_info=True)
        return {"error": str(e), "exists": False, "has_product_masks": False, "masks": []}

@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=5000)