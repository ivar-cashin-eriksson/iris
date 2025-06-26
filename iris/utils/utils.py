import json
import os
from glob import glob
from pathlib import Path
from typing import Union
from urllib.parse import urlparse, urlunparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pycocotools import mask
from tqdm import tqdm

# TODO: Move or delete relevant util functions

def apply_mask_overlay(image_np, masks, alpha=0.5):
    """Apply color masks over the image."""
    mask_overlay = np.zeros_like(image_np, dtype=np.uint8)
    num_masks = masks.shape[0]
    colors = plt.cm.get_cmap("hsv", num_masks + 1)
    mask_colors = [np.array(colors(i)[:3]) * 255 for i in range(num_masks)]

    for i, mask in enumerate(masks):
        mask_overlay[mask > 0] = mask_colors[i]

    return (alpha * mask_overlay + (1 - alpha) * image_np).astype(np.uint8)


def draw_bounding_boxes(image_np, bboxes, thickness=2):
    """Draw bounding boxes on the image."""
    if bboxes is not None:
        for bbox in bboxes:
            x, y, w, h = bbox
            cv2.rectangle(
                image_np, (x, y), (x + w, y + h), (255, 0, 0), thickness=thickness
            )
    return image_np


def plot_separate_masks(masks, mask_quality, point_coords, bboxes):
    """Plot each mask separately with bounding boxes."""
    num_masks = masks.shape[0]
    fig, axes = plt.subplots(1, num_masks, figsize=(12, 6))
    colors = plt.cm.get_cmap("hsv", num_masks + 1)
    mask_colors = [np.array(colors(i)[:3]) * 255 for i in range(num_masks)]

    for i, mask in enumerate(masks):
        colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        colored_mask[mask > 0] = mask_colors[i]
        colored_mask_with_bbox = draw_bounding_boxes(
            colored_mask, [bboxes[i]] if bboxes is not None else None, thickness=8
        )

        axes[i].imshow(colored_mask_with_bbox)
        title = f"Mask {i+1}"
        if mask_quality is not None:
            title += f"\nQuality: {mask_quality[i]:.2f}"
        axes[i].set_title(title, fontsize=10)

        if point_coords is not None:
            axes[i].plot(point_coords[i][0][0], point_coords[i][0][1], ".", color="w")

        axes[i].axis("off")
    plt.tight_layout()
    plt.show()


def plot_masks_on_image(
    image,
    masks,
    bboxes=None,
    bboxes_on_overlay=True,
    mask_quality=None,
    point_coords=None,
    alpha=0.5,
):
    """
    Overlay segmentation masks on a PIL image and display them side by side with bounding boxes.
    """
    image_np = np.array(image)
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()

    blended = apply_mask_overlay(image_np, masks, alpha)
    if bboxes_on_overlay:
        blended = draw_bounding_boxes(blended, bboxes)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image_np)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(blended)
    axes[1].set_title("Mask Overlay")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

    plot_separate_masks(masks, mask_quality, point_coords, bboxes)


def crop_object_from_bbox(image, bbox):
    """
    Crops an object from an image using the provided bounding box in XYWH format.

    :param image: The original image (NumPy array).
    :param bbox: The bounding box (X, Y, W, H).
    :return: Cropped object as an image array.
    """
    x, y, w, h = bbox  # Extract values from XYWH format
    cropped_img = image[y : y + h, x : x + w]  # Crop the region
    return cropped_img


def encode_rle(binary_mask):
    rle = mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode(
        "utf-8"
    )  # Convert bytes to string for JSON compatibility
    return rle


# Function to decode RLE back to binary mask
def decode_rle(rle):
    rle["counts"] = rle["counts"].encode("utf-8")  # Convert string back to bytes
    return mask.decode(rle)


# Function to process images and generate COCO-style dataset
def generate_coco_dataset(image_folder, output_json, sam_model):
    """
    Processes images, extracts masks using SAM2, and saves results in COCO JSON format.
    """

    # COCO JSON structure
    coco_dataset = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "object"}],  # Single category for now
    }

    image_files = glob(
        os.path.join(image_folder, "**", "*.jpg"), recursive=True
    ) + glob(os.path.join(image_folder, "**", "*.png"), recursive=True)
    annotation_id = 1

    for image_id, image_path in enumerate(tqdm(image_files, desc="Processing Images")):
        # Load image
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        relative_path = os.path.relpath(image_path, image_folder)

        # Add image metadata to COCO dataset
        coco_dataset["images"].append(
            {
                "id": image_id,
                "file_name": relative_path,
                "width": width,
                "height": height,
            }
        )

        # Process image with SAM2
        masks = sam_model.generate(image)

        for mask_data in masks:
            segmentation = encode_rle(mask_data["segmentation"])
            area = mask_data["area"]
            bbox = mask_data["bbox"]
            predicted_iou = mask_data["predicted_iou"]
            point_coords = mask_data["point_coords"]
            stability_score = mask_data["stability_score"]
            crop_box = mask_data["crop_box"]

            # Store annotation
            coco_dataset["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "segmentation": segmentation,
                    "area": area,
                    "bbox": bbox,
                    "predicted_iou": predicted_iou,
                    "point_coords": point_coords,
                    "stability_score": stability_score,
                    "crop_box": crop_box,
                }
            )

            annotation_id += 1

    # Save COCO JSON file
    with open(output_json, "w") as f:
        json.dump(coco_dataset, f, indent=4)

    print(f"COCO dataset saved to {output_json}")


# Function to read COCO JSON and reconstruct masks and metadata for a specified image
def reconstruct_image_data(json_file, image_id, image_library_path=None):
    """
    Reads a COCO-style JSON file and reconstructs all mask and metadata for a given image ID.
    """
    with open(json_file, "r") as f:
        coco_data = json.load(f)

    image_info = next(
        (img for img in coco_data["images"] if img["id"] == image_id), None
    )
    if not image_info:
        raise ValueError(f"Image ID {image_id} not found in the dataset.")

    annotations = [
        ann for ann in coco_data["annotations"] if ann["image_id"] == image_id
    ]

    reconstructed_data = []
    for ann in annotations:
        rle_mask = ann["segmentation"]
        binary_mask = decode_rle(rle_mask)

        reconstructed_data.append(
            {
                "segmentation": binary_mask,
                "area": ann["area"],
                "bbox": ann["bbox"],
                "predicted_iou": ann["predicted_iou"],
                "point_coords": ann["point_coords"],
                "stability_score": ann["stability_score"],
                "crop_box": ann["crop_box"],
            }
        )

    image_path = image_info["file_name"]  # Extract image path from JSON
    if image_library_path is not None:
        image_path = os.path.join(image_library_path, image_path)
    image = Image.open(image_path)
    image = np.array(image.convert("RGB"))

    return image_info, reconstructed_data, image


def normalize_image_url(url: str) -> str:
    """
    Normalize image URL by removing query parameters that don't affect the core image.
    This ensures consistent storage and prevents duplicates for the same image 
    with different parameters like imwidth, imheight, quality, etc.
    
    Args:
        url (str): The original image URL
        
    Returns:
        str: The normalized URL without query parameters
        
    Examples:
        >>> normalize_image_url("https://media.cos.com/image.jpg?imwidth=657")
        "https://media.cos.com/image.jpg"
        >>> normalize_image_url("https://example.com/photo.png?width=800&height=600")
        "https://example.com/photo.png"
    """
    if not url:
        return url
        
    parsed = urlparse(url)
    # Remove query parameters - keeping only the base URL
    normalized = urlunparse(
        (
            parsed.scheme,
            parsed.netloc, 
            parsed.path,
            '',  # params
            '',  # query - removed
            ''   # fragment
        )
    )
    return normalized
