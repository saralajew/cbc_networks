# -*- coding: utf-8 -*-
"""To be able to run the provided scripts, a small modification needs to be
made to the keras preprocessing package. In line with most literature on
ImageNet we used a custom rescaling of the images, this rescaling option is
not available through the standard keras Library.

To be exact, the modification needs to be made in
`keras_preprocessing.image.utils`. Here the function
`load_img` need to be replaced with the one provided here.
"""

def load_img(path, grayscale=False, color_mode='rgb', target_size=None,
             interpolation='nearest'):
    """Loads an image into PIL format.

    # Arguments
        path: Path to image file.
        color_mode: One of "grayscale", "rgb", "rgba". Default: "rgb".
            The desired image format.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", and "bicubic".
            If PIL version 1.1.3 or newer is installed, "lanczos" is also
            supported. If PIL version 3.4.0 or newer is installed, "box" and
            "hamming" are also supported. By default, "nearest" is used.

    # Returns
        A PIL Image instance.

    # Raises
        ImportError: if PIL is not available.
        ValueError: if interpolation method is not supported.
    """

    def imagenet_resize(img, size):
        old_size = img.size

        if not size[0] == size[1]:
            raise ValueError('Size must be square for the custom imagenet resize option')
        ratio = size[0] / min(old_size)  # This assumes a square size

        ## Resize shortest side to the correct size, while maintaining the aspect ratio
        new_height = round(old_size[0] * ratio)
        new_width = round(old_size[1] * ratio)
        img = img.resize((new_height, new_width), pil_image.ANTIALIAS)

        ## Center crop the image
        width_dif = (img.size[0] - size[0]) // 2
        height_dif = (img.size[1] - size[1]) // 2

        img = img.crop((width_dif, height_dif, width_dif + size[0], height_dif + size[1]))

        return img

    if grayscale is True:
        warnings.warn('grayscale is deprecated. Please use '
                      'color_mode = "grayscale"')
        color_mode = 'grayscale'
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    img = pil_image.open(path)
    if color_mode == 'grayscale':
        if img.mode != 'L':
            img = img.convert('L')
    elif color_mode == 'rgba':
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
    elif color_mode == 'rgb':
        if img.mode != 'RGB':
            img = img.convert('RGB')
    else:
        raise ValueError('color_mode must be "grayscale", "rgb", or "rgba"')
    if target_size is not None:
        width_height_tuple = (target_size[1], target_size[0])
        if img.size != width_height_tuple:
            if interpolation == "custom_imagenet":
                img = imagenet_resize(img, width_height_tuple)
            elif interpolation in _PIL_INTERPOLATION_METHODS:
                resample = _PIL_INTERPOLATION_METHODS[interpolation]
                img = img.resize(width_height_tuple, resample)
            else:
                raise ValueError(
                    'Invalid interpolation method {} specified. Supported '
                    'methods are {}'.format(
                        interpolation,
                        ", ".join(_PIL_INTERPOLATION_METHODS.keys())))
    return img
