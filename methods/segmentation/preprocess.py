import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import imutils
import shutil

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


if __name__ == "__main__":
    undoPreprocess()

    