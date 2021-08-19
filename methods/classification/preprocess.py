import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import imutils
import shutil
import platform

def preprocess():
    current = os.getcwd()

    path = os.path.join(current, "classify_data")
    os.chdir(path)

    # check if preprocess has occured before, else do nothing
    if not os.path.exists(os.path.join(path, "TEST")):

        # split the data by train/val/test
        for label in os.listdir():

            if not label.startswith('.'):

                cur = os.listdir(os.path.join(path, label))
                IMG_NUM = len(cur)

                for (n, file) in enumerate(cur):
                    img = os.path.join(path, label, file)

                    # set five images of each label aside for testing
                    if n < 5:
                        newPath = os.path.join(path, "TEST", label, label + "_tumor_" + str(n+1) + ".jpg")
                        os.makedirs(os.path.dirname(newPath), exist_ok=True)
                        shutil.copy(img, newPath)

                    # set 80% of remaining files aside for training
                    elif n < 0.8*IMG_NUM:
                        newPath = os.path.join(path, "TRAIN", label, label + "_tumor_" + str(n+1) + ".jpg")
                        os.makedirs(os.path.dirname(newPath), exist_ok=True)
                        shutil.copy(img, newPath)

                    # rest for validation
                    else:
                        newPath = os.path.join(path, "VALIDATE", label, label + "_tumor_" + str(n+1) + ".jpg")
                        os.makedirs(os.path.dirname(newPath), exist_ok=True)
                        shutil.copy(img, newPath)
    os.chdir(current)


# function that crops images of tumors 
def crop_imgs(set_name, add_pixels_value=0):
    """
    Finds the extreme points on the image and crops the rectangular out of them
    """
    set_new = []
    for img in set_name:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # threshold the image, then perform a series of erosions +
        # dilations to remove any small regions of noise
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # find contours in thresholded image, then grab the largest one
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        # find the extreme points
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        ADD_PIXELS = add_pixels_value
        new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
        set_new.append(new_img)


    # img = cv2.imread('../input/brain-mri-images-for-brain-tumor-detection/brain_tumor_dataset/yes/Y108.jpg')
    # img = cv2.resize(
    #             img,
    #             dsize=IMG_SIZE,
    #             interpolation=cv2.INTER_CUBIC
    #         )
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # # threshold the image, then perform a series of erosions +
    # # dilations to remove any small regions of noise
    # thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    # thresh = cv2.erode(thresh, None, iterations=2)
    # thresh = cv2.dilate(thresh, None, iterations=2)

    # # find contours in thresholded image, then grab the largest one
    # cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = imutils.grab_contours(cnts)
    # c = max(cnts, key=cv2.contourArea)

    # # find the extreme points
    # extLeft = tuple(c[c[:, :, 0].argmin()][0])
    # extRight = tuple(c[c[:, :, 0].argmax()][0])
    # extTop = tuple(c[c[:, :, 1].argmin()][0])
    # extBot = tuple(c[c[:, :, 1].argmax()][0])

    # # add contour on the image
    # img_cnt = cv2.drawContours(img.copy(), [c], -1, (0, 255, 255), 4)

    # # add extreme points
    # img_pnt = cv2.circle(img_cnt.copy(), extLeft, 8, (0, 0, 255), -1)
    # img_pnt = cv2.circle(img_pnt, extRight, 8, (0, 255, 0), -1)
    # img_pnt = cv2.circle(img_pnt, extTop, 8, (255, 0, 0), -1)
    # img_pnt = cv2.circle(img_pnt, extBot, 8, (255, 255, 0), -1)

    # # crop
    # ADD_PIXELS = 0
    # new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()

    # plt.figure(figsize=(15,6))
    # plt.subplot(141)
    # plt.imshow(img)
    # plt.xticks([])
    # plt.yticks([])
    # plt.title('Step 1. Get the original image')
    # plt.subplot(142)
    # plt.imshow(img_cnt)
    # plt.xticks([])
    # plt.yticks([])
    # plt.title('Step 2. Find the biggest contour')
    # plt.subplot(143)
    # plt.imshow(img_pnt)
    # plt.xticks([])
    # plt.yticks([])
    # plt.title('Step 3. Find the extreme points')
    # plt.subplot(144)
    # plt.imshow(new_img)
    # plt.xticks([])
    # plt.yticks([])
    # plt.title('Step 4. Crop the image')
    # plt.show()

    return np.array(set_new)


if __name__ == "__main__":
    preprocess()