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

async def _get_base_response(url: str, db):
    """Helper function to get base response structure and image document"""
    image_doc = db.image_metadata.find_one({"original_url": url})
    
    if not image_doc:
        return None, {
            "exists": False,
            "has_product_detections": False,
            "detections": []
        }

    has_product_detections = False
    if "localizations" in image_doc and image_doc["localizations"]:
        has_product_detections = any(
            "product_hash" in localization 
            for localization in image_doc["localizations"]
        )

    response = {
        "exists": True,
        "has_product_detections": has_product_detections,
        "detections": []
    }
    
    return image_doc, response

def _get_product_image_url(product, db):
    """Helper function to get the first product image URL"""
    if product.get("image_hashes") and len(product["image_hashes"]) > 0:
        first_image_hash = product["image_hashes"][0]
        image_meta = db.image_metadata.find_one({"image_hash": first_image_hash})
        if image_meta:
            return image_meta.get("original_url")
    return None

@app.get("/get-detections")
async def check_url(url: str = Query(...), db_name: str = Query(...)):
    """
    Check if a URL exists in the image_metadata collection and return localization information.
    """
    try:
        db = client[db_name]
        image_doc, response = await _get_base_response(url, db)
        if not image_doc:
            return response
        
        if "localizations" in image_doc and image_doc["localizations"]:
            for localization in image_doc["localizations"]:
                if "localization_point" in localization and "product_predictions" in localization:
                    product_predictions = localization["product_predictions"]
                    prediction_hash = max(product_predictions.items(), key=lambda x: x[1])[0]

                    product = db.products.find_one({"product_hash": prediction_hash})
                    if product:
                        product_image_url = _get_product_image_url(product, db)
                        response["detections"].append({
                            "point": localization["localization_point"],
                            "product_predictions": [
                                {
                                    "product_url": product.get("url"),
                                    "product_title": product.get("title"),
                                    "product_price": product.get("price"),
                                    "product_image": product_image_url
                                }
                            ]
                        })
        
        logger.info(f"Checked URL {url} in database {db_name}: exists={response['exists']}, has_product_detections={response['has_product_detections']}")
        return response
    
    except Exception as e:
        logger.error(f"Error checking URL: {str(e)}", exc_info=True)
        return {"error": str(e), "exists": False, "has_product_detections": False, "detections": []}

@app.get("/get-detections-all-predictions")
async def check_url(url: str = Query(...), db_name: str = Query(...)):
    """
    Check if a URL exists in the image_metadata collection and return all prediction information.
    """
    try:
        db = client[db_name]
        image_doc, response = await _get_base_response(url, db)
        if not image_doc:
            return response
        
        if "localizations" in image_doc and image_doc["localizations"]:
            for localization in image_doc["localizations"]:
                if "localization_point" in localization and "product_predictions" in localization:
                    localisation_predictions = []
                    for prediction_hash, score in localization["product_predictions"].items():
                        product = db.products.find_one({"product_hash": prediction_hash})
                        if product:
                            product_image_url = _get_product_image_url(product, db)
                            localisation_predictions.append({
                                "score": score,
                                "product_url": product.get("url"),
                                "product_title": product.get("title"),
                                "product_price": product.get("price"),
                                "product_image": product_image_url
                            })
                    response["detections"].append(
                        {
                            "point": localization["localization_point"],
                            "product_predictions": localisation_predictions
                        }
                    )
        
        logger.info(f"Checked URL {url} in database {db_name}: exists={response['exists']}, has_product_detections={response['has_product_detections']}")
        return response
    
    except Exception as e:
        logger.error(f"Error checking URL: {str(e)}", exc_info=True)
        return {"error": str(e), "exists": False, "has_product_detections": False, "detections": []}

@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=5000)