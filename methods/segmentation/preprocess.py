import numpy as np
import os
import shutil
import nibabel as nb
from matplotlib.cm import get_cmap
from imageio import mimwrite
from skimage.transform import resize

from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

def preprocess(num_augment_gen=20):
    current = os.getcwd()

    path = os.path.join(current, "BRATS_data")
    os.chdir(path)

    # check if preprocess has occured before, else do nothing
    if not os.path.exists(os.path.join(path, "TEST")):

        # split the data by train/val/test
        for label in os.listdir():

            if not label.startswith('.'):

                cur = os.listdir(os.path.join(path, label))
                num_img = len(cur)

                for (n, file) in enumerate(cur):
                    img = os.path.join(path, label, file)

                    # set five images of each label aside for testing
                    if n < 5:
                        newPath = os.path.join(path, "TEST", label, label + "_tumor_" + str(n+1) + ".jpg")
                        os.makedirs(os.path.dirname(newPath), exist_ok=True)
                        shutil.copy(img, newPath)

                    # set 80% of remaining files aside for training
                    elif n < 0.8 * num_img:
                        newPath = os.path.join(path, "TRAIN", label, label + "_tumor_" + str(n+1) + ".jpg")
                        os.makedirs(os.path.dirname(newPath), exist_ok=True)
                        shutil.copy(img, newPath)

                    # rest for validation
                    else:
                        newPath = os.path.join(path, "VALIDATE", label, label + "_tumor_" + str(n+1) + ".jpg")
                        os.makedirs(os.path.dirname(newPath), exist_ok=True)
                        shutil.copy(img, newPath)
    
        if num_augment_gen:
            # increase datatset by adding augmented images
            for label in os.listdir(os.path.join(path, "TRAIN")):
                directory = os.path.join(path, "TRAIN", label)
                augment(directory, num_augment_gen, directory)

    os.chdir(current)


def undoPreprocess():
    current = os.getcwd()
    path = os.path.join(current, "classify_data")
    os.chdir(path)

    for dir in os.listdir():
        if dir in ("TEST", "TRAIN", "VALIDATE"):
            shutil.rmtree(os.path.join(path, dir))
        elif dir in ("no", "yes"): 
            curDir = os.path.join(path, dir)
            for file in os.listdir(curDir):
                if "aug" in file:
                    os.remove(os.path.join(curDir, file))

    os.chdir(current)


# augment the data and then save to dataset
def augment(directory, num_samples, save_directory):
    # increase size of dataset using augmented images
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        brightness_range=[0.5, 1.5],
        horizontal_flip=True,
        vertical_flip=True
    )

    for file in os.listdir(directory):
        img = load_img(os.path.join(directory, file))
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)   
        i = 0

        for batch in datagen.flow(x, batch_size=1, save_to_dir=save_directory, save_prefix='aug_'+file[:-4], save_format='jpg'):
            i+=1
            if i > num_samples:
                break


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

    out_img /= out_img.max()  # scale image values between 0-1

    # Resize image by the following factor
    if size != 1:
        out_img = resize(out_img, [int(size * maximum)] * 3)

    maximum = int(maximum * size)

    return out_img, maximum


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



if __name__ == "__main__":
    undoPreprocess()

    