from typing import List, Tuple, Optional, Union
import io
import base64
import numpy as np
from PIL import Image, ImageDraw
from IPython.display import display, HTML
import colorsys
import textwrap
from iris.utils.image_utils import convert_mask_format, convert_image_format
from iris.data_pipeline.mongodb_manager import MongoDBManager

# -----------------------------
# Shared Utilities
# -----------------------------

def generate_distinct_color(idx: int) -> Tuple[int, int, int]:
    hsv = (idx * 0.618033988749895 % 1.0, 0.5, 0.95)
    return tuple((int(c * 255) for c in colorsys.hsv_to_rgb(*hsv)))

def image_to_base64(img: Union[np.ndarray, Image.Image]) -> str:
    img = convert_image_format(img, target_format="pil")
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def render_metadata_block(metadata: List[Tuple[str, str]], inline: bool=False) -> str:
    block = ""
    for label, content in metadata:
        if inline:
            block += f"""
            <div style='margin-bottom: 5px; color: black;'>
                <strong>{label}</strong> {content}
            </div>
            """
        else:
            block += f"""
            <div style="margin-bottom: 5px; color: black;">
                <strong>{label}</strong>
                <div style="margin-left: 20px; overflow-wrap: break-word; word-break: break-word;">
                    {content}
                </div>
            </div>
            """
    return block

# -----------------------------
# Image Rendering
# -----------------------------

def render_mask_card(mask_data: dict, idx: int) -> str:
    binary_mask = convert_mask_format(mask_data["segmentation"], target_format="binary")
    bbox = mask_data.get("bbox", [0, 0, 1, 1])
    x, y, w, h = bbox
    color = generate_distinct_color(idx)

    mask_rgb = Image.new("RGB", (binary_mask.shape[1], binary_mask.shape[0]), color=(0, 0, 0))
    overlay = Image.fromarray(np.uint8(binary_mask) * 255, mode="L")
    color_layer = Image.new("RGB", mask_rgb.size, color)
    mask_rgb.paste(color_layer, mask=overlay)

    draw = ImageDraw.Draw(mask_rgb)
    draw.rectangle([x, y, x + w, y + h], outline="white", width=2)

    # Add a small cross at the sampled point
    point_coords = mask_data.get("point_coords", [[None, None]])[0]
    if point_coords[0] is not None and point_coords[1] is not None:
        px, py = point_coords
        cross_size = 10
        draw.line([(px - cross_size, py), (px + cross_size, py)], fill="white", width=2)
        draw.line([(px, py - cross_size), (px, py + cross_size)], fill="white", width=2)

    mask_base64 = image_to_base64(mask_rgb)

    metadata = [
        ("Mask Hash:", mask_data["localization_hash"]),
        ("Crop Box:", mask_data["crop_box"]),
        ("Bounding Box:", f"[{x}, {y}, {w}, {h}]"),
        ("Mask Area:", mask_data["area"]),
        ("Point Coords:", f"[{point_coords[0]}, {point_coords[1]}]"),
        ("Predicted IoU:", mask_data["predicted_iou"]),
        ("Stability Score:", mask_data["stability_score"]),
    ]

    return f"""
    <div style='display: flex; align-items: start; gap: 10px;'>
        <img src="data:image/png;base64,{mask_base64}" style="width: 100px; height: auto; display: block;">
        <div style='color: black; font-size: 11px;'>
            {render_metadata_block(metadata, inline=True)}
        </div>
    </div>
    """


def render_mask_grid(masks: list, columns: int, sort_key: str, reverse: bool) -> str:
    sorted_indices = sorted(
        range(len(masks)),
        key=lambda i: masks[i][sort_key],
        reverse=reverse
    )

    mask_cards = "\n".join(
        render_mask_card(masks[idx], idx)
        for idx in sorted_indices
    )

    return f"""
    <div style='margin-top: 40px;'>
        <div style='display: grid; grid-template-columns: repeat({columns}, minmax(0, 1fr)); gap: 20px;'>
            {mask_cards}
        </div>
    </div>
    """

def overlay_masks_on_image(img: Image.Image, masks: list) -> Image.Image:
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    for idx, mask_data in enumerate(masks):
        binary_mask = convert_mask_format(mask_data["segmentation"], target_format="binary")
        color = generate_distinct_color(idx) + (90,)  # Add alpha channel
        mask_img = Image.fromarray(np.uint8(binary_mask) * 255, mode="L")
        color_layer = Image.new("RGBA", img.size, color)
        overlay.paste(color_layer, mask=mask_img)
    return Image.alpha_composite(img, overlay)

def display_image_summary(
    mongodb_manager: MongoDBManager,
    image_hash: str,
    columns: int = 3,
    sort_key: str = "predicted_iou",
    reverse: bool = True,
    show_masks: bool = True
):
    image_data = mongodb_manager.get_collection(
        mongodb_manager.mongodb_config.image_metadata_collection
    ).find_one({"image_hash": image_hash})

    if image_data is None:
        raise ValueError(f"No image found with hash: {image_hash}")

    # Make main images
    with Image.open(image_data['local_path']).convert("RGBA") as img:
        if show_masks and "localizations" in image_data:
            masked_img = overlay_masks_on_image(img.copy(), image_data["localizations"])
        else:
            masked_img = img.copy()

        img_str = image_to_base64(img)
        masked_str = image_to_base64(masked_img)

    # Image block
    image_display_block = f"""
        <img src="data:image/png;base64,{img_str}" style="max-width: 200px; height: auto;">
    """
    if show_masks and "localizations" in image_data:
        image_display_block += f"""
        <img src="data:image/png;base64,{masked_str}" style="max-width: 200px; height: auto;">
        """

    # Metadata block
    metadata = [
        ("Image Hash:", image_data['image_hash']),
        ("Num Masks:", len(image_data['localizations']) if 'localizations' in image_data else 'Image not segmented.'),
        ("Local Path:", f'<a href="{image_data["local_path"]}" style="color: #0066cc;">{image_data["local_path"]}</a>'),
        ("Original URL:", f'<a href="{image_data["original_url"]}" style="color: #0066cc;">{image_data["original_url"]}</a>'),
        ("Source Product:", image_data['source_product']),
        ("Created At:", image_data['created_at'])
    ]

    # Mask block
    mask_cards_html = ""
    if show_masks and "localizations" in image_data:
        mask_cards_html = render_mask_grid(
            masks=image_data["localizations"],
            columns=columns,
            sort_key=sort_key,
            reverse=reverse
        )

    # Assemble HTML
    html = f"""
    <div style="
        background-color: white;
        padding: 20px;
        border-radius: 5px;
        max-width: 1000px;
        margin: 0 auto;
        font-family: sans-serif;
        color: black;
    ">
        <div style="display: flex; gap: 20px; align-items: flex-start;">
            {image_display_block}
            <div style="
                background: #f5f5f5;
                padding: 20px;
                border-radius: 5px;
                overflow: hidden;
                overflow-wrap: break-word;
                word-break: break-word;
                flex-grow: 1;
            ">
            {render_metadata_block(metadata)}
            </div>
        </div>
        {mask_cards_html}
    </div>
    """

    display(HTML(html))


# -----------------------------
# Product Rendering and Summary
# -----------------------------

def render_product_images(image_hashes: list, mongodb_manager: MongoDBManager, max_image_size: int, columns: int) -> str:
    tiles = []
    for image_hash in image_hashes:
        image_data = mongodb_manager.get_collection(
            mongodb_manager.mongodb_config.image_metadata_collection
        ).find_one({"image_hash": image_hash})
        img = Image.open(image_data['local_path'])
        if max(img.size) > max_image_size:
            ratio = max_image_size / max(img.size)
            new_size = tuple(int(dim * ratio) for dim in img.size)
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        img_str = image_to_base64(img)
        tiles.append(f"""
        <div style="text-align: center;">
            <img src="data:image/png;base64,{img_str}" style="max-width: 100%; height: auto;">
            <div style="font-size: 10px; color: #666;">{image_hash}</div>
        </div>
        """)
    html = f"""
    <div style="
        display: grid;
        grid-template-columns: repeat({columns}, 1fr);
        gap: 10px;
        margin-bottom: 20px;
    ">
        {''.join(tiles)}
    </div>
    """

    return html

def display_product_summary(
    mongodb_manager: MongoDBManager,
    product_hash: str,
    max_image_size: int = 300,
    columns: int = 3
):
    product_data = mongodb_manager.get_collection(
        mongodb_manager.mongodb_config.product_collection
    ).find_one({"product_hash": product_hash})
    if product_data is None:
        raise ValueError(f"No product found with hash: {product_hash}")

    # Metadata block
    metadata = [
        ("Price:", product_data['price']),
        ("Description:", product_data['description']),
        ("URL:", f'<a href="{product_data["url"]}" style="color: #0066cc;">{product_data["url"]}</a>'),
        ("Hash:", product_data['product_hash']),
        ("Created At:", product_data['created_at'])
    ]

    # Image block
    product_images_html = render_product_images(
        product_data['image_hashes'],
        mongodb_manager=mongodb_manager,
        max_image_size=max_image_size,
        columns=columns
    )

    html = f"""
    <div style="
        background-color: white;
        padding: 20px;
        border-radius: 5px;
        max-width: 1000px;
        margin: 0 auto;
        font-family: sans-serif;
        color: black;
    ">
        <h2 style="text-align: center; margin-bottom: 20px; color: black;">{product_data['title']}</h2>
        {product_images_html}
        <div style="background: #f5f5f5; padding: 20px; border-radius: 5px; color: black;">
            {render_metadata_block(metadata)}
        </div>
    </div>
    """

    display(HTML(html))

# -----------------------------
# Print Utilities
# -----------------------------

def print_image_summary(
    mongodb_manager: MongoDBManager,
    image_hash: str,
    total_width: int = 60,
    first_column_width: int = 20
):
    image_data = mongodb_manager.get_collection(
        mongodb_manager.mongodb_config.image_metadata_collection
    ).find_one({"image_hash": image_hash})
    if image_data is None:
        raise ValueError(f"No image found with hash: {image_hash}")

    separator = "-" * total_width
    print(separator)
    print(f"{' ' * int((total_width - len('Image Summary')) / 2)}Image Summary")
    print(separator)
    print(f"{'Image Hash:':<{first_column_width}} {image_data['image_hash']}")
    print(f"{'Num Masks:':<{first_column_width}} {len(image_data.get('localizations', [])) if 'localizations' in image_data else 'Image not segmented.'}")
    print(f"{'HTML Location:':<{first_column_width}} {image_data['html_location']}")
    print(f"{'Local Path:':<{first_column_width}} {image_data['local_path']}")
    print(f"{'Original URL:':<{first_column_width}} {image_data['original_url']}")
    print(f"{'Source Product:':<{first_column_width}} {image_data['source_product']}")
    print(f"{'Created At:':<{first_column_width}} {image_data['created_at']}")
    print(separator)

def print_product_summary(
    mongodb_manager: MongoDBManager,
    product_hash: str,
    total_width: int = 60,
    first_column_width: int = 20
):
    product_data = mongodb_manager.get_collection(
        mongodb_manager.mongodb_config.product_collection
    ).find_one({"product_hash": product_hash})
    if product_data is None:
        raise ValueError(f"No product found with hash: {product_hash}")

    separator = "-" * total_width
    print(separator)
    print(f"{' ' * int((total_width - len('Product Summary')) / 2)}Product Summary")
    print(separator)
    print(f"{'Title:':<{first_column_width}} {product_data['title']}")
    print(f"{'Product Hash:':<{first_column_width}} {product_data['product_hash']}")
    print(f"{'Price:':<{first_column_width}} {product_data['price']}")

    wrapped = textwrap.fill(product_data['description'], width=total_width - first_column_width)
    indented = "\n".join(f"{' ' * (1 + first_column_width)}{line}" for line in wrapped.splitlines())
    print(f"{'Description:':<{first_column_width}} {indented.strip()}")

    print(f"{'URL:':<{first_column_width}} {product_data['url']}")
    print(f"{'Num Images:':<{first_column_width}} {len(product_data['image_hashes'])}")
    print(f"{'Created At:':<{first_column_width}} {product_data['created_at']}")
    print(separator)

