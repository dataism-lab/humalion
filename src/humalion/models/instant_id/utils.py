import numpy as np
from PIL import Image
from PIL.Image import Resampling


def resize_img(
    input_image: Image,
    max_side=1280,
    min_side=1024,
    size=None,
    pad_to_max_side=False,
    mode: Resampling = Image.BILINEAR,
    base_pixel_number=64,
) -> Image:
    """
    Resizes an input image according to specified parameters.

    Args:
        input_image (Image): The input image to be resized.
        max_side (int): The maximum side length for the resized image. Default is 1280.
        min_side (int): The minimum side length for the resized image. Default is 1024.
        size (tuple): The desired size (width, height) for the resized image. Default is None.
        pad_to_max_side (bool): Whether to pad the image to the maximum side length. Default is False.
        mode: The interpolation mode to use during resizing. Default is Image.BILINEAR.
        base_pixel_number (int): The base number of pixels for resizing. Default is 64.

    Returns:
        Image: The resized image.

    """
    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio * w), round(ratio * h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio * w), round(ratio * h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y : offset_y + h_resize_new, offset_x : offset_x + w_resize_new] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image
