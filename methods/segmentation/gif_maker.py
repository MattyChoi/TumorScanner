import numpy as np
import os
import nibabel as nb
from imageio import mimwrite
from skimage.transform import resize


def parse_filename(filepath):
    
    path = os.path.normpath(filepath)
    dirname = os.path.dirname(path)
    filename = path.split(os.sep)[-1]
    basename, ext = filename.split(os.extsep, 1)
    return dirname, basename, ext


def prepare_image(data, size=1):
    # Load NIfTI file
    # data = nb.load(filename).get_fdata()

    # create data array template with zeros to make the shape isometric
    maximum = np.max(data.shape)

    out_img = np.zeros([maximum] * 3)

    a, b, c = data.shape
    x, y, z = (list(data.shape) - maximum) / -2

    out_img[int(x): int(x) + a, int(y): int(y) + b, int(z): int(z) + c] = data

    out_img *= 255 / out_img.max()  # scale image values between 0-1

    # Resize image by the following factor
    if size != 1:
        out_img = resize(out_img, [int(size * maximum)] * 3)

    maximum = int(maximum * size)

    return out_img.astype(np.uint8), maximum


def create_mosaic_normal(out_img, maximum):

    new_img = np.array(
        [np.hstack((
            np.hstack(
                (
                    np.flip(out_img[i, :, :], 1).T,
                    np.flip(out_img[:, maximum - i - 1, :], 1).T)),
                    np.flip(out_img[:, :, maximum - i - 1], 1).T
                )
            )
            for i in range(maximum)
        ]
    )

    return new_img


def write_gif_normal(filename, size=1, fps=18):
    # load data
    data = nb.load(filename).get_fdata()

    # Load NIfTI and put it in right shape
    out_img, maximum = prepare_image(data, size)

    # Create output mosaic
    new_img = create_mosaic_normal(out_img, maximum)

    # Figure out extension
    ext = '.{}'.format(parse_filename(filename)[2])

    # Write gif file
    mimwrite(filename.replace(ext, '.gif'), new_img,
             format='gif', fps=int(fps * size))
