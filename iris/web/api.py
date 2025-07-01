from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from iris.config.data_pipeline_config_manager import DataPipelineConfigManager
from iris.data_pipeline.mongodb_manager import MongoDBManager
from iris.models.product import Product
from iris.models.image import Image
from iris.models.localization import Localization   
from iris.utils.log import logger
from iris.utils.utils import normalize_image_url

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MongoDB manager
config_manager = DataPipelineConfigManager()
mongodb_manager = MongoDBManager(config_manager.mongodb_config)


async def _get_base_response(
        url: str, 
        db_name: str
    ) -> tuple[list[Localization], dict[str, any]]:
    """
    Helper function to get base response structure and image document

    Note: Currently the MondoDBManager does not allow for dynamic database selection,
    so we assume the correct database is already set in the MongoDBManager configuration.
    """
    # Normalize the URL to handle different query parameters
    normalized_url = normalize_image_url(url)
    
    with mongodb_manager as db:
        image: Image = db.find_one(
            collection_name=db.config.image_metadata_collection,
            query={"url": normalized_url},
        )
    
    if image is None:
        return None, {
            "exists": False,
            "has_product_detections": False,
            "detections": []
        }

    has_product_detections = False
    localizations: list[Localization] = []
    if len(image.localization_hashes) > 0:
        localizations = db.find_all(
            collection_name=db.config.localization_collection,
            query={"hash": {"$in": image.localization_hashes}},
        )

        has_product_detections = any(
            len(localization.product_predictions) > 0
            for localization in localizations
        )

    response = {
        "exists": True,
        "has_product_detections": has_product_detections,
        "detections": []
    }
    
    return localizations, response


@app.get("/get-detections")
async def check_url(
        url: str = Query(...), 
        db_name: str = Query(...)
    ) -> dict[str, any]:
    """
    Check if a URL exists in the database and return product detections if any.

    Args:
        url (str): The URL to check.
        db_name (str): The name of the database to query. 
                       NOTE: The database name is currently not used and the 
                             name is inferred from the MongoDBManager 
                             configuration instead.

    Returns:
        dict: A dictionary containing the detection results.
    """
    try:
        localizations, response = await _get_base_response(url, db_name)
        
        if not localizations or response["has_product_detections"] is False:
            return response
        
        with mongodb_manager as db:
            for localization in localizations:
                product_hash = max(
                    localization.product_predictions.items(), key=lambda x: x[1]
                )[0]

                product: Product = db.find_one(
                    collection_name=db.config.product_collection,
                    query={"hash": product_hash}
                )

                with mongodb_manager as db:
                    product_image: Image = db.find_one(
                        collection_name=db.config.image_metadata_collection,
                        query={"hash": product.image_hashes[0]}
                    )
                response["detections"].append({
                    "point": localization.point,
                    "product_predictions": [
                        {
                            "product_url": product.url,
                            "product_title": product.metadata.get("title", ""),
                            "product_price": product.metadata.get("price", ""),
                            "product_image": product_image.url
                        }
                    ]
                })

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