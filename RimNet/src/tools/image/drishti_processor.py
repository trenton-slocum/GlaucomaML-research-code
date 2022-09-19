'''
Processing the Drishti images.

We'll generate two different rims.
1. Using the whitest area (most agree on)
2. Using the largest area

We're assuming all images are .png format
We're assuming all images have a corresponding GT segmentation folder
We're assuming the cup segmentation folder contains 'cupseg' in the file name

Some images do not have a disc and cup, we'll skip those and make a note

The AvgBoundary folder has a file *_diskCenter.txt which we can use for cropping
We'll assume the files have no header and the coordinates are seperated by space
'''

import glob
import os
import pathlib

import cv2
import matplotlib.pyplot as plt
import pandas as pd

error_checking = False
no_disc = pd.DataFrame(columns=(['IMAGE', 'MISSING_CUP', 'MISSING_DISC']))
no_disc_path = "..//no_disc.csv"

# Image processing values
crop_x_size = 1200
crop_y_size = 1200
level = "Level_1"

thresholds = {"Level_1": 252, "Level_2": 187, "Level_3": 124, "Level_4": 60}
thresh, maxValue = thresholds[level], 255

# Where to read images
path_images_raw = "../images/Drishti/Drishti-GS1_files/Test/Images/*.png"
path_segmentation_folders = "../images/Drishti/Drishti-GS1_files/Test/Test_GT/"

# Where to save images
path_raw_processed_save = "../images/Drishti_" + level + "_Ready/Raw/"
path_rim_processed_save = "../images/Drishti_" + level + "_Ready/Rim/"
path_cup_processed_save = "../images/Drishti_" + level + "_Ready/Cup/"
path_disc_processed_save = "../images/Drishti_" + level + "_Ready/Disc/"

# Create save directories
for directory in [
        path_raw_processed_save, path_rim_processed_save,
        path_cup_processed_save, path_disc_processed_save
]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Read in all the images
images_raw = glob.glob(path_images_raw)

for i, img_raw_path in enumerate(images_raw):
    print(str(i + 1), len(images_raw), img_raw_path, sep=" : ")
    # Find the folder for the current raw image
    image_name = pathlib.Path(img_raw_path).stem
    segmentation_folder = os.path.join(path_segmentation_folders, image_name)

    # Does it exist? If not, throw error and exit loop
    if not os.path.isdir(segmentation_folder):
        print("Segmentation folder not found for ", image_name, sep="")
        print(f'Tried to find: {segmentation_folder}')
        break

    # If it does exist, we'll assume the folder has the following subfolders
    # AvgBoundary : text files with information
    # SoftMap : segmentation images for cup and disc

    # The AvgBoundary folder has a file *_diskCenter.txt which we can use for cropping
    # We'll assume the files have no header and the coordinates are seperated by space
    disc_center = pd.read_csv(os.path.join(segmentation_folder, "AvgBoundary",
                                           image_name + "_diskCenter.txt"),
                              sep=" ",
                              header=None)

    # Read in the cup and disc from the SoftMap folder, assuming folder only has two png files
    # Assuming the cup has cupseg and the disc does not
    segmentation_images = glob.glob(
        os.path.join(segmentation_folder, "SoftMap", "*.png"))
    img_cup = [
        segmentation_images for segmentation_images in segmentation_images
        if 'cupseg' in segmentation_images
    ]
    img_disc = [
        segmentation_images for segmentation_images in segmentation_images
        if not 'cupseg' in segmentation_images
    ]

    if len(img_disc) > 1:
        print("Too many discs found for", image_name)
        break
    if len(img_cup) > 1:
        print("Too many cups found for", image_name)
        break

    if len(img_disc) == 0 or len(img_cup) == 0:
        no_disc = no_disc.append(
            {
                'IMAGE': image_name,
                'MISSING_CUP': len(img_cup),
                'MISSING_DISC': len(img_disc)
            },
            ignore_index=True)
        continue

    # We have our three images, let's read em in
    img_raw = cv2.imread(img_raw_path)
    img_cup = cv2.imread(img_cup[0], cv2.IMREAD_GRAYSCALE)
    img_disc = cv2.imread(img_disc[0], cv2.IMREAD_GRAYSCALE)

    if error_checking:
        fig, ax = plt.subplots(nrows=1, ncols=3)
        plt.subplot(1, 3, 1)
        plt.imshow(img_raw[:, :, ::-1])
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.imshow(img_cup, cmap='gray')
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.imshow(img_disc, cmap='gray')
        plt.axis('off')

    # Threshold the cup and disc to get only the white area (most agree upon area)
    # We'll use simple binary thresholding
    th, img_cupthreshold = cv2.threshold(img_cup, thresh, maxValue,
                                         cv2.THRESH_BINARY)
    th, img_disc_threshold = cv2.threshold(img_disc, thresh, maxValue,
                                           cv2.THRESH_BINARY)
    rim = img_disc_threshold - img_cupthreshold

    if error_checking:
        fig, ax = plt.subplots(nrows=1, ncols=3)
        plt.subplot(1, 3, 1)
        plt.imshow(img_cupthreshold, cmap='gray')
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.imshow(img_disc_threshold, cmap='gray')
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.imshow(rim, cmap='gray')
        plt.axis('off')

    # Crop around the center
    start_row = disc_center.iloc[0][0] - crop_y_size // 2
    end_row = disc_center.iloc[0][0] + crop_y_size // 2
    start_col = disc_center.iloc[0][1] - crop_x_size // 2
    end_col = disc_center.iloc[0][1] + crop_x_size // 2
    img_raw_cropped = img_raw[start_row:end_row, start_col:end_col]
    rim_cropped = rim[start_row:end_row, start_col:end_col]
    cup_cropped = img_cupthreshold[start_row:end_row, start_col:end_col]
    disc_cropped = img_disc_threshold[start_row:end_row, start_col:end_col]

    if error_checking:
        fig, ax = plt.subplots(nrows=1, ncols=2)
        plt.subplot(1, 2, 1)
        plt.imshow(img_raw_cropped[:, :, ::-1])
        plt.subplot(1, 2, 2)
        plt.imshow(rim_cropped, cmap='gray')

        plt.imshow(cup_cropped, cmap='gray')
        plt.imshow(disc_cropped, cmap='gray')

    # Save
    cv2.imwrite(os.path.join(path_raw_processed_save, image_name + ".jpg"),
                img_raw_cropped)
    cv2.imwrite(os.path.join(path_rim_processed_save, image_name + ".jpg"),
                rim_cropped)
    cv2.imwrite(os.path.join(path_cup_processed_save, image_name + ".jpg"),
                cup_cropped)
    cv2.imwrite(os.path.join(path_disc_processed_save, image_name + ".jpg"),
                disc_cropped)

no_disc.to_csv(no_disc_path, index=False)
