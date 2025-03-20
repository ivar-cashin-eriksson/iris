from product_handler import ProductHandler
from mongodb_manager import MongoDBManager


def main():
    # Test URLs
    urls = [
        "https://pasnormalstudios.com/dk/products/off-race-logo-hoodie",
        "https://pasnormalstudios.com/dk/products/off-race-cotton-twill-pants-limestone"
    ]

    # CSS selectors for product data
    product_filters = {
        "title": "body > main > div > div > div:nth-child(1) > section.block-wrapper.space-y-5.lg\:hidden > div > div > div.text-xl.font-medium.leading-5",
        "price": "body > main > div > div > div:nth-child(1) > section.block-wrapper.space-y-5.lg\:hidden > div > div > div.hidden.md\:block > div > div > button > div > div",
        "color": "body > main > div > div > div:nth-child(1) > section.block-wrapper.space-y-5.lg\:hidden > div > div > button > div > div.flex.items-center.gap-1 > span",
        "num_colorways": "body > main > div > div > div:nth-child(1) > section.block-wrapper.space-y-5.lg\:hidden > div > div > button > div > div.text-\[9px\].uppercase.opacity-70 > span:nth-child(1)",
        "description": "body > main > div > div > div:nth-child(1) > section.block-wrapper.space-y-5.lg\:hidden > div > div > div:nth-child(4) > div > div > p",
        "made_in": "#radix-\:r7d\: > div.flex.h-full.w-full.flex-col.border-t.px-5.pt-6.md\:px-8 > div.py-12.md\:py-10 > div.flex.w-full.items-center.justify-start.border-t.border-gray-light.py-6 > div > div.flex.w-56.text-xl.font-medium.md\:text-2xl > div",
        "product_details": "#radix-\:r7d\: > div.flex.h-full.w-full.flex-col.border-t.px-5.pt-6.md\:px-8 > div.py-12.md\:py-10 > div.flex.w-full.items-center.justify-start.border-y.border-gray-light.py-6 > div > div.flex.w-56.text-xs.leading-4"
    }

    # CSS selector for product images
    image_selector = "body > main > div > div > div:nth-child(1) > section.block-wrapper.page-offset-notification.relative.bg-\[\#f1f1f1\].lg\:h-screen.lg\:pt-0 > div.relative.transition-all.lg\:h-screen > div.relative.hidden.lg\:block.h-full"

    # MongoDB connection string
    connection_string = "mongodb+srv://test:test@iris-cluster.andes.mongodb.net/?retryWrites=true&w=majority&appName=iris-cluster"
    
    # Initialize MongoDB connection
    mongo_manager = MongoDBManager(connection_string)

    # Create product handler
    product_handler = ProductHandler(
        filters=product_filters,
        mongo_manager=mongo_manager,
        image_selector=image_selector
    )

    # Process products
    print("Starting product processing...")
    product_handler.process_products(urls)

    # Demonstrate the relationships
    print("\nDemonstrating product-image relationships:")
    
    # Get all products
    products = mongo_manager.find_all('products')
    for product in products:
        print(f"\nProduct: {product['title']}")
        print(f"Product Hash: {product['product_hash']}")
        print(f"URL: {product['url']}")
        print(f"Number of images: {len(product['image_hashes'])}")
        
        # Get image details for each image hash in the product
        print("Images:")
        for image_hash in product['image_hashes']:
            image = mongo_manager.find_one('image_metadata', {'image_hash': image_hash})
            if image:
                print(f"  - Hash: {image['image_hash']}")
                print(f"    Local Path: {image['local_path']}")
                print(f"    Original URL: {image['original_url']}")

    # Close MongoDB connection
    mongo_manager.close()


if __name__ == "__main__":
    main() 