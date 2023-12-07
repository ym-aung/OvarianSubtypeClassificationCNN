from pathlib import Path
import os
from PIL import Image

ORIGINAL_THUMBNAIL_FOLDER_PATH = Path(".\\train_thumbnails")
PROCESSED_THUMBNAIL_FOLDER_PATH = ".\\processed_train_thumbnails"

RESIZE_WIDTH = 64
RESIZE_HEIGHT = 64

# Resize all images to the same dimensions
for thumbnail_path in ORIGINAL_THUMBNAIL_FOLDER_PATH.iterdir():
    img = Image.open(os.path.join(".", thumbnail_path))

    new_img = img.resize((RESIZE_WIDTH, RESIZE_HEIGHT))

    new_img.save(os.path.join(PROCESSED_THUMBNAIL_FOLDER_PATH, os.path.basename(thumbnail_path)))
