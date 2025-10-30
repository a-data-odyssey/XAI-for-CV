import requests
from io import BytesIO

from PIL import Image

def save_image(url, filename, crop_to_square=True,resize=(224,224)):
    """Save image from url"""

    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    img = Image.open(BytesIO(response.content)).convert("RGB")

    if crop_to_square:
        width, height = img.size
        new_edge = min(width, height)
        left = (width - new_edge) / 2
        top = (height - new_edge) / 2
        right = (width + new_edge) / 2
        bottom = (height + new_edge) / 2
        img = img.crop((left, top, right, bottom))

    if resize:
        img = img.resize(resize)

    img.save(filename)
