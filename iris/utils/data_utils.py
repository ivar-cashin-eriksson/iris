import textwrap
import matplotlib.pyplot as plt
import requests


# ============================================================
#                          Product
# ============================================================

def plot_product_summary(product_data):
    """Plot the product image and summary information.

    Args:
        product_data (dict): A dictionary containing product information.
    """
    image_url = product_data['image_url']
    product_name = product_data['name']
    product_price = product_data['price']

    # Fetch and display the image
    response = requests(image_url)

    plt.imshow(img)
    plt.title(f"{product_name} - ${product_price}")
    plt.axis("off")
    plt.show()

def print_product_summary(product_data, total_width=60, first_column_width=20):
    """Print the product summary information in a readable format.

    Args:
        product_data (dict): A dictionary containing product metadata.
    """
    # Define the separator
    separator = "-" * total_width

    # Print the header with a separator
    print(separator)
    title = "Product Summary"
    print(f"{' ' * int((total_width - len(title))/2) + title}")
    print(separator)

    # Print the data in a formatted way
    print(f"{'Product Hash:':<{first_column_width}} {product_data['product_hash']}")
    print(f"{'Title:':{first_column_width}} {product_data['title']}")
    print(f"{'Price:':<{first_column_width}} {product_data['price']}")
    
    # Wrap the description for better readability
    description = product_data['description']
    wrapped_description = textwrap.fill(description, width=total_width - first_column_width)

    # Indent the wrapped description correctly
    indented_description = "\n".join(f"{' ' * (1 + first_column_width)}{line}" for line in wrapped_description.splitlines())
    print(f"{'Description:':<{first_column_width}} {indented_description.strip()}")  # Remove leading spaces from the first line


    print(f"{'URL:':<{first_column_width}} {product_data['url']}")
    print(f"{'Num Images:':<{first_column_width}} {len(product_data['image_hashes'])}")
    print(f"{'Created At:':<{first_column_width}} {product_data['created_at']}")

    # Print the footer with a separator
    print(separator)


# ============================================================
#                          Image
# ============================================================

def plot_image_sumamry():
    pass

def print_image_summary(image_data, total_width=60, first_column_width=20):
    """Print the image summary information in a readable format.

    Args:
        image_data (dict): A dictionary containing image metadata.
    """
    # Define the separator
    separator = "-" * total_width

    # Print the header with a separator
    print(separator)
    title = "Image Summary"
    print(f"{' ' * int((total_width - len(title)) / 2) + title}")
    print(separator)

    # Print the data in a formatted way
    print(f"{'Image Hash:':<{first_column_width}} {image_data['image_hash']}")
    print(f"{'Source Product:':<{first_column_width}} {image_data['source_product']}")

    # If the image has been segmented
    if "masks" in image_data:
        print(f"{'Num Masks:':<{first_column_width}} {len(image_data['masks'])}")
    else:
        print(f"{'Num Masks:':<{first_column_width}} Image not segmented.")

    print(f"{'Local Path:':<{first_column_width}} {image_data['local_path']}")
    print(f"{'HTML Location:':<{first_column_width}} {image_data['html_location']}")
    print(f"{'Original URL:':<{first_column_width}} {image_data['original_url']}")
    print(f"{'Created At:':<{first_column_width}} {image_data['created_at']}")

    # Print the footer with a separator
    print(separator)