# -*- coding: utf-8 -*-
"""
Script to generate a DDLS grade for each fundus image in the dataset.
First calls the segmentation model on the image and calculates minimum
rim to disc ratio or angle of missing rim.

This information is then combined with the predicted disc size to generate a
DDLS grade.

We are not making all images RE for the disc size since vertical flip augmentation was used
in training the model

Steps:
    Rim/Disc Ratio:
        Load image
        Resize (down) image to rim_model specifications
        Apply CLAHE
        Apply the rim_model to get the rim segmentation
        Create the mask from the segmentation
        Apply post processing
        resize (up) to original dimensions
        Extract the disc and cup for cropping the image for disc model
        Find the narrowest rim/disc ratio
    Disc Size:
        Crop around center of disc 1200x1200
        resize (down) to disc_model specifications
        Apply model to get disc size
    Calculate DDLS

"""

import glob
import os
import pathlib

import cv2
import numpy as np
import pandas as pd
import segmentation_models as sm
import tensorflow as tf
import tools.calculators.DDLS_Calculator as ddls
import tools.image.post_processing as pp
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


###################################################
def load_image(img_path, img_width, img_height, show=False):
    img = image.load_img(img_path, target_size=(img_width, img_height))

    img_tensor = image.img_to_array(img)  # (height, width, channels)
    img_tensor = np.expand_dims(
        img_tensor, axis=0
    )  # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    #img_tensor /= 255.                                     # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(np.squeeze(img_tensor) / 255)
        plt.axis('off')
        plt.show()

    return img_tensor


def apply_clahe(image):
    # Assuming color is in RGB since read in by tf.image.decode_jpeg

    img_hist = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_hist[:, :, 0] = clahe.apply(img_hist[:, :, 0])

    # convert the YUV image back to RGB format
    image = cv2.cvtColor(img_hist, cv2.COLOR_YUV2RGB)

    return image


def create_mask(pred_mask):
    """ Converts continuous 0 to 1 outputs into either 0 or 1."""
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    pred_mask = np.array(pred_mask)
    pred_mask = np.squeeze(pred_mask)
    return pred_mask

def create_mask_multiclass(pred_mask):
    """ Converts a W,H,N mask to W,H with integer labels"""
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


###################################################


###################################################
# Applying models
def apply_disc_model(img_path, img_raw):
    img = load_image(img_path,
                     disc_model_img_width,
                     disc_model_img_height,
                     show=False)
    img = apply_clahe(np.squeeze(img.astype(np.uint8)))
    img = np.expand_dims(img, 0).astype(np.float32)

    if error_checking:
        plt.imshow(img[0].astype(np.uint32))

    # Find the disc, use this to center and crop
    disc_model_predicted_disc = disc_model.predict(img)

    if disc_model_predicted_disc.shape[-1] == 1:
        # This output only has one class in it.
        # We need to convert back to 2 output layers for the
        # rest of the processing
        new_disc = np.ones(
            disc_model_predicted_disc.shape) - disc_model_predicted_disc

        disc_model_predicted_disc = np.concatenate(
            [new_disc, disc_model_predicted_disc], axis=-1)

    disc_model_predicted_disc = create_mask(disc_model_predicted_disc)
    disc_model_predicted_disc = 255 * disc_model_predicted_disc.astype(
        np.uint8)
    disc_model_predicted_disc = pp.find_largest_connected_component(
        disc_model_predicted_disc, dilate=False, erode=False)

    # Convert image back to original dimensions
    resize_width = img_raw.shape[1]
    resize_height = img_raw.shape[0]
    disc_model_predicted_disc = cv2.resize(disc_model_predicted_disc,
                                           (resize_width, resize_height),
                                           interpolation=cv2.INTER_LINEAR)

    return (disc_model_predicted_disc)


def apply_disc_size(predicted_disc, img_raw, results, error_checking=False):
    # Crop around the center of the cup
    contours, _ = cv2.findContours(predicted_disc, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    cnt = contours[0]

    M = cv2.moments(cnt)
    x = int(round(M["m10"] / M["m00"]))
    y = int(round(M["m01"] / M["m00"]))

    y_range_start = max([y - crop_size[0], 0])
    y_range_end = min([y + crop_size[0], img_raw.shape[0]])
    x_range_start = max([x - crop_size[1], 0])
    x_range_end = min([x + crop_size[1], img_raw.shape[1]])

    img = img_raw[y_range_start:y_range_end, x_range_start:x_range_end, :]

    if error_checking:
        plt.imshow(img)

    # Resize to model specifications
    #img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (disc_size_img_width, disc_size_img_height))
    img = np.expand_dims(img, axis=0)

    print('image range info')
    print(img.max())
    print(img.min())

    if error_checking:
        plt.imshow(img[0])
        plt.imshow(
            np.expand_dims(tf.image.resize(
                img_raw, (disc_size_img_width, disc_size_img_height)),
                           axis=0)[0])

    # Load image, resize, expand dim for model, predict size
    disc_size = disc_size_model.predict(img)
    disc_size = np.argmax(disc_size) + 1
    results['DISC_SIZE'] = disc_size

    print(f'disc size: {disc_size}')

    if error_checking:
        print(str(i + 1),
              len(raw_images),
              os.path.basename(img_path),
              disc_size,
              sep=' : ')


def apply_rim_model(img_path):
    # Load image, resize, apply CLAHE, expand dim for model
    img = load_image(img_path, rim_img_width, rim_img_height, show=False)
    img = apply_clahe(np.squeeze(img.astype(np.uint8)))
    img = np.expand_dims(img, 0).astype(np.float32)

    # Apply the rim_model
    predicted_rim = rim_model.predict(img)

    if predicted_rim.shape[-1] == 1:
        # This output only has one class in it.
        # We need to convert back to 2 output layers for the
        # rest of the processing
        new_rim = np.ones(predicted_rim.shape) - predicted_rim

        predicted_rim = np.concatenate([new_rim, predicted_rim], axis=-1)

    predicted_rim = create_mask_multiclass(predicted_rim)
    predicted_rim = np.array(predicted_rim, dtype=np.uint8)

    return (predicted_rim)


def rim_connected_check(predicted_rim, results, error_checking=False):
    '''
    At this point, the model either predicted a connect rim or broken rim
    We'll check this by:
        Find the largest connected component, to get rid of any small noise
        Also, by doing this, rims disconnected >1 are reduced to a single disconnect
        contour hierarchy, if there is only 1 level it is disconnected, else there is a "doughnut" shape

    If the rim is disconnected, we'll find the TWO largest contours, it is possible for a rim
    to be disconnected in two locations.
    TO DO: check with Dr. Caprioli is maximum two is reasonable
    Then find the total disconnected angle

    If the rim is connected, we'll find the single largest contour, find the RDR

    predicted_rim = cv2.imread("..\\\\images\\broken_rim_tests\\mask_scaled\\1.3.6.1.4.1.29565.1.4.0.27655.1453808400.445496.MASK.jpg")
    predicted_rim = cv2.cvtColor(predicted_rim, cv2.COLOR_BGR2GRAY)
    predicted_rim = cv2.threshold(predicted_rim, 200, 255, cv2.THRESH_BINARY)[1]
    predicted_rim_disconnected_test = predicted_rim
    predicted_rim = rim_truth

    check:  '..\\images\\3.28.22 Broken-Whole Rim Ready\\train\\raw\\UCLA_1.3.6.1.4.1.29565.1.4.0.27655.1453781634.436925.jpg'
    '''

    predicted_rim_disconnected_test = pp.find_largest_connected_component(
        predicted_rim, dilate=False, erode=False)
    if error_checking:
        plt.imshow(predicted_rim_disconnected_test, cmap='gray')

    _, hierarchy = cv2.findContours(predicted_rim_disconnected_test,
                                    cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    if hierarchy.shape[1] >= 2:
        results['TYPE'] = "Connected"
    elif hierarchy.shape[1] == 1:
        results['TYPE'] = "Disconnected"
    else:
        results['TYPE'] = "Error"
        print("Error in heiarchy")


def find_rim_disc_ratio(predicted_rim,
                        predicted_disc,
                        predicted_cup,
                        img_raw,
                        results,
                        img_path,
                        error_checking=False):

    # Find the narrowest rim
    display_rim, df_rim_diameters = ddls.find_diameters(
        cv2.imread(img_path)[:, :, ::-1],
        predicted_rim,
        line_width=10,
        circle_diam=1,
        cup_thickness=10,
        disc_thickness=10,
        rim_thickness=15)
    rim_disc_ratio = df_rim_diameters.loc[np.argmin(
        df_rim_diameters['rim/disc'])]

    if False:
        plt.imshow(display_rim)


    results['RIM_DIAM'] = rim_disc_ratio['rim_diam']
    results['DISC_DIAM'] = rim_disc_ratio['disc_diam']
    results['CUP_DIAM'] = rim_disc_ratio['cup_diam']
    results['RIM_DISC_RATIO'] = rim_disc_ratio['rim/disc']

    rim_disc_ratio = rim_disc_ratio['rim/disc']

    return (display_rim, df_rim_diameters)


###################################################

###################################################
print('reading models')
rim_model = load_model("../models/20220426-062458-Rim-multiclass_best",
                       custom_objects={'iou_score': sm.metrics.iou_score})
_, rim_img_width, rim_img_height, rim_img_channels = rim_model.layers[
    0].input_shape[0]

#disc_size_model = load_model("../models/disc_size_tuned_best-augsv2-fixed_scale")
disc_size_model = load_model("../models/disc_size_tuned_best-more-pics")
_, disc_size_img_width, disc_size_img_height, disc_size_img_channels = disc_size_model.layers[
    0].input_shape[0]

# Used to extract the disc for disconnected rims, so we can estimate the center
# We can also use for cropping around for the DiscNet
disc_model = load_model(
    "../models/20220131-230633-dm-new-backbone_tuned_mp_best",
    custom_objects={'iou_score': sm.metrics.iou_score})
_, disc_model_img_width, disc_model_img_height, disc_model_img_channels = disc_model.layers[
    0].input_shape[0]
print('done')
###################################################

###################################################
# os.chdir("..//")
df_results_empty = pd.DataFrame(columns=('IMAGE', 'TYPE', 'DISC_SIZE',
                                         'DISC_SIZE_FROM', 'RIM_DISC_RATIO',
                                         'ANGLES', 'ANGLE_SUM', 'DDLS_SCORE',
                                         'DISC_AREA', 'CUP_AREA', 'RIM_AREA',
                                         'RIM_DIAM', 'DISC_DIAM', 'CUP_DIAM', 'GROUP'))
results_save_path = '..//files_output//DDLS_Pipeline_Output__20220426-062458-Rim-multiclass_best.csv'
results_true_save_path = '..//files_output//DDLS_Pipeline_Output_True__20220426-062458-Rim-multiclass_best.csv'

df_results_empty.to_csv(results_save_path, mode='w', header=True, index=False)

df_results_empty.to_csv(results_true_save_path,
                        mode='w',
                        header=True,
                        index=False)

###################################################

###################################################
crop_size = (1000, 1000)

#############

raw_images = glob.glob(
    str(pathlib.Path('../images/broken_rim_tests//Raw//*.jpg')))
df_images = {
    "RAW":
    raw_images,
    "MASKS": [
        '../images/broken_rim_tests//mask_scaled//' + pathlib.Path(i).stem +
        ".MASK.jpg" for i in raw_images
    ],
    "DDLS_IMAGE": [
        '../images/broken_rim_tests//ddls//' + os.path.basename(i)
        for i in raw_images
    ],
    "DDLS_IMAGE_STEPS": [
        '../images/broken_rim_tests//ddls_steps//' + os.path.basename(i)
        for i in raw_images
    ]
}

df_images = pd.DataFrame(df_images)
#############

#############
raw_image_path = '..//images//4.22.22.Raw.Fundus//'
image_segments_path = '..//images//4.24.22 Broken-Whole Rim Ready//'

save_image_path = '../images/20220426-062458-Rim-multiclass_best/ddls/'
save_image_path_steps = '../images/20220426-062458-Rim-multiclass_best/ddls_steps/'

df_images = pd.DataFrame()

print('reading in raw images')
for split in ['train', 'test', 'val']:
    curr_path = raw_image_path + '//' + split
    raw_images = glob.glob(
        str(pathlib.Path(f'{image_segments_path}//{split}//raw//*.png')))

    temp = {
        "RAW":
        raw_images,
        "MASKS": [
            f'{image_segments_path}//{split}//mask//' + os.path.basename(i)
            for i in raw_images
        ],
        "DDLS_IMAGE": [
            f'{image_segments_path}//{split}//' + os.path.basename(i)
            for i in raw_images
        ],
        "DDLS_IMAGE_STEPS": [
            f'{image_segments_path}//{split}//' + os.path.basename(i)
            for i in raw_images
        ],
        "GROUP":
        split
    }
    temp = pd.DataFrame(temp)
    df_images = df_images.append(temp)

df_images['GROUP'].unique()
print('done')

#df_images['MASKS'] = [os.path.dirname(i) + "//"+ pathlib.Path(i).stem+".png" if "broken" in os.path.basename(i) else i for i in df_images['MASKS'] ]


#############
def iou(result1, result2):
    '''Intersection Over Union.

    Parameters
    ----------
    result1 : numpy.ndarray
    result2 : numpy.ndarray

    Returns
    -------
    list
        Calcaulted IoU for each class

    '''
    # 1 or 0. 1 means rim
    value = 1
    filtered_res1 = (result1 == value)
    filtered_res2 = (result2 == value)

    intersection = np.logical_and(filtered_res1, filtered_res2)
    union = np.logical_or(filtered_res1, filtered_res2)
    iou_score = np.sum(intersection) / np.sum(union)

    # Return rim iou, which we assume is the smaller one
    return iou_score


###################
print('creating dirs')
# Create directories if needed
if 'DDLS_IMAGE' in df_images.columns:
    for folder in df_images['DDLS_IMAGE'].apply(
            lambda x: os.path.dirname(x)).unique():
        if not os.path.exists(folder):
            os.makedirs(folder)

if 'DDLS_IMAGE_STEPS' in df_images.columns:
    for folder in df_images['DDLS_IMAGE_STEPS'].apply(
            lambda x: os.path.dirname(x)).unique():
        if not os.path.exists(folder):
            os.makedirs(folder)
###################
print('done creating dirs')

print('reading a csv')
rim_net_demo = pd.read_csv("..//files//clinician_train_val_disc_size - lourdes_broken.csv")
rim_net_demo = rim_net_demo[rim_net_demo['DISC_SIZE_AGREE'] >= 1]
rim_net_demo['TYPE'].unique()
FILE_ONLY = True
print('done')

###################
# TO DO, use alternative to iterrows
error_checking = False
print('iterating')
print(len(df_images))

# Images subset
for i, index_row in enumerate(df_images.iterrows()):
    row = index_row[1]
    index = index_row[0]
    img_path = row['RAW']
    img_path = glob.glob(str(pathlib.Path(f'{raw_image_path}//{pathlib.Path(img_path).stem}.png')))[0]

    if FILE_ONLY:
        if (rim_net_demo['IMAGE_STEM'] == pathlib.Path(img_path).stem).any():
            #print("Found")
            true_info = rim_net_demo[rim_net_demo['IMAGE_STEM'] == pathlib.Path(img_path).stem]
        else:
            continue

    # Subset
    '''
    if not( pathlib.Path(img_path).stem in ['1.3.6.1.4.1.29565.1.4.0.18030.1453912853.63898',
                                       '1.3.6.1.4.1.29565.1.4.0.18030.1453914157.64324',
                                       '1.3.6.1.4.1.29565.1.4.0.18030.1453928630.68899',
                                       '1.3.6.1.4.1.29565.1.4.0.18030.1453940486.72614',
                                       '1.3.6.1.4.1.29565.1.4.0.18030.1453943000.73437']):
        continue
    '''
    if not( pathlib.Path(img_path).stem in ['1.3.6.1.4.1.29565.1.4.0.18030.1453920640.66377',
                                            '1.3.6.1.4.1.29565.1.4.0.18030.1453978539.84422',
                                            '1.3.6.1.4.1.29565.1.4.0.18030.1453989685.87854',
                                            '1.3.6.1.4.1.29565.1.4.0.27655.1453804085.444142',
                                            'AXIS01_27242_410418_2015111216004563648d561a049b638e2_3',
                                            'AXIS01_54810_669073_2016101310465518461c7e0be40312efa_5']):
        continue





    #print(index)
    #break
    # if i < 860:
    #     continue

    print(str(i + 1),
          df_images.shape[0],
          os.path.basename(img_path),
          sep=' : ')

    ##################################
    results = {
        'IMAGE': "",
        'TYPE': "",
        'DISC_SIZE': "",
        'DISC_SIZE_FROM': "Model",
        'RIM_DISC_RATIO': "",
        'ANGLES': "",
        'ANGLE_SUM': "",
        'DDLS_SCORE': "",
        'DISC_AREA': "",
        'CUP_AREA': "",
        'RIM_AREA': "",
        'RIM_DIAM': "",
        'DISC_DIAM': "",
        'CUP_DIAM': ""
    }

    if 'GROUP' in df_images.columns:
        results['GROUP'] = row['GROUP']

    # if 'broken' in img_path:
    #     continue

    results['IMAGE'] = os.path.basename(img_path)
    results['ANGLES'] = 0
    results['ANGLE_SUM'] = 0
    ##################################

    ##################################
    predicted_disc = []
    display_rim = []
    predicted_rim = []
    predicted_rim_scaled = []
    ##################################

    ##################################
    img_raw = tf.io.read_file(img_path)
    img_raw = tf.image.decode_jpeg(img_raw)
    #img_raw = tf.image.convert_image_dtype(img_raw, tf.float32)
    ##################################

    '''
    ##################################
    # Disc Model
    # Load image, resize, apply CLAHE, expand dim for model
    disc_model_predicted_disc = apply_disc_model(img_path, img_raw)

    if error_checking:
        plt.imshow(disc_model_predicted_disc, cmap='gray')
    ##################################
    '''

    ##################################
    # Rim Model
    # We'll either extract a rim/disc or a broken angle total
    predicted_multiclass = apply_rim_model(img_path)
    ##################################

    ##################################
    # Extract rim, disc, cup
    resize_width = img_raw.shape[1]
    resize_height = img_raw.shape[0]
    predicted_multiclass_scaled = cv2.resize(predicted_multiclass, (resize_width, resize_height), interpolation=cv2.INTER_NEAREST)

    predicted_rim = 255 * np.where(predicted_multiclass_scaled == 1, 1, 0).astype(np.uint8)
    predicted_cup = 255 * np.where(predicted_multiclass_scaled == 2, 1, 0).astype(np.uint8)
    predicted_disc = predicted_rim + predicted_cup

    if error_checking:
        plt.imshow(predicted_multiclass, cmap='gray')
        plt.imshow(predicted_multiclass_scaled, cmap='gray')
        plt.imshow(predicted_rim, cmap='gray')
        plt.imshow(predicted_cup, cmap='gray')
        plt.imshow(predicted_disc, cmap='gray')

    results['DISC_AREA'] = np.sum(predicted_disc == 255)
    results['CUP_AREA'] = np.sum(predicted_cup == 255)
    results['RIM_AREA'] = np.sum(predicted_rim == 255)
    ##################################

    ##################################
    # Check if there is any hiearchy
    # results is a mutable dictionary, passed by referenced
    rim_connected_check(predicted_rim, results)

    if results['TYPE'] == "Connected":
        display_rim, df_rim_diameters = find_rim_disc_ratio(predicted_rim,
                                    predicted_disc,
                                    predicted_cup, img_raw, results, img_path)

    elif results['TYPE'] == "Disconnected":
        display_rim, angles = ddls.find_angle(predicted_rim, predicted_disc, cv2.imread(img_path)[:, :, ::-1])

        results['ANGLES'] = angles
        results['ANGLE_SUM'] = np.sum(angles)

        # If the angle is 0, that means we have a bad center, this shouldn't be considered
        if results['ANGLE_SUM'] == 0:
            results['DDLS_SCORE'] = 0

    if error_checking:
        plt.imshow(display_rim)

    ##################################

    #################
    # Disc Size
    apply_disc_size(predicted_disc, img_raw, results)
    #################

    #################
    # DDLS Score
    if results['DDLS_SCORE'] != 0:
        results['DDLS_SCORE'] = ddls.ddls_score(
            disc_size=results['DISC_SIZE'],
            ratio=results['RIM_DISC_RATIO'],
            angle=results['ANGLE_SUM'])
    #################

    #################
    # Save images for each step
    # print(f'trying {row["MASKS"]}')
    # print(os.path.exists(row["MASKS"]))
    if 'DDLS_IMAGE_STEPS' in df_images.columns and os.path.exists(row["MASKS"]):
        results_true = {
            'IMAGE': "",
            'TYPE': "",
            'DISC_SIZE': "",
            'DISC_SIZE_FROM': "Clinician",
            'RIM_DISC_RATIO': "",
            'ANGLES': "",
            'ANGLE_SUM': "",
            'DDLS_SCORE': "",
            'DISC_AREA': "",
            'CUP_AREA': "",
            'RIM_AREA': "",
            'RIM_DIAM': "",
            'DISC_DIAM': "",
            'CUP_DIAM': "",
            'GROUP': row['GROUP']
        }

        results_true['IMAGE'] = os.path.basename(img_path)
        results_true['ANGLES'] = 0
        results_true['ANGLE_SUM'] = 0

        results_true['DISC_SIZE'] = true_info.iloc[0]['DISC_SIZE_MODE']

        predicted_disc_true = []
        rim_truth = []

        #####################
        # Multitruuth truth
        multiclass_truth = cv2.cvtColor(cv2.imread(row["MASKS"]), cv2.COLOR_RGB2BGR)

        if multiclass_truth.shape[1] != resize_width:
            multiclass_truth = cv2.resize(multiclass_truth, (resize_width, resize_height), interpolation=cv2.INTER_NEAREST)


        rim_truth = multiclass_truth[:,:,0]
        cup_truth = multiclass_truth[:,:,2] - multiclass_truth[:,:,0]
        disc_truth = rim_truth + cup_truth

        if False:
            plt.imshow(multiclass_truth)
            plt.imshow(rim_truth, cmap='gray')
            plt.imshow(cup_truth, cmap='gray')
            plt.imshow(disc_truth, cmap='gray')


        results_true['DISC_AREA'] = np.sum(disc_truth == 255)
        results_true['CUP_AREA'] = np.sum(cup_truth == 255)
        results_true['RIM_AREA'] = np.sum(rim_truth == 255)
        #####################

        #####################
        # IoU
        '''
        rim_model_small = 255 * np.where(predicted_multiclass == 1, 1, 0).astype(np.uint8)
        resized_truth = cv2.resize(rim_truth, (224, 224))
        rim_iou = iou(np.clip(rim_model_small, 0, 1),
                      np.clip(resized_truth, 0, 1))
        print(f'Rim IoU: {rim_iou}')
        results['RIM_IOU'] = rim_iou
        '''
        #####################

        rim_connected_check(rim_truth, results_true)
        # RDR or Angle
        if results_true['TYPE'] == "Connected":
            display_rim_true, _ = find_rim_disc_ratio(rim_truth,
                                        disc_truth,
                                        cup_truth, img_raw, results_true, img_path)
        elif results_true['TYPE'] == "Disconnected":
            display_rim_true, angles = ddls.find_angle(rim_truth, disc_truth, cv2.imread(img_path)[:, :, ::-1])

            results_true['ANGLES'] = angles
            results_true['ANGLE_SUM'] = np.sum(angles)

            # If the angle is 0, that means we have a bad center, this shouldn't be considered
            if results_true['ANGLE_SUM'] == 0:
                results_true['DDLS_SCORE'] = 0

        if False:
            plt.imshow(display_rim_true)

        # DDLS Score
        if results_true['DDLS_SCORE'] != 0:
            results_true['DDLS_SCORE'] = ddls.ddls_score(
                disc_size=results_true['DISC_SIZE'],
                ratio=results_true['RIM_DISC_RATIO'],
                angle=results_true['ANGLE_SUM'])

        # Save the results
        df_results = df_results_empty.copy()
        df_results = df_results.append(results_true, ignore_index=True)
        df_results.to_csv(results_true_save_path,
                          mode='a',
                          header=False,
                          index=False)

        mono_font = {'fontname':'monospace'}
        #########################
        # Image RDR RADAR
        nrows = 1
        ncols = 3
        i = 0
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols)

        # Raw
        i += 1
        plt.subplot(nrows, ncols, i)
        plt.imshow(cv2.imread(img_path)[:, :, ::-1])
        plt.axis('off')
        plt.title('')

        # Clinician
        rim_area_disc_area = results_true['RIM_AREA'] / results_true['DISC_AREA']
        if results_true['TYPE'] == "Connected":
            title = f" Rim/Disc: {np.round(results_true['RIM_DISC_RATIO'],2):.2f}\n      ARW: {np.round(rim_area_disc_area,2):.2f}"
        else:
            title = f"Rim Angle: {np.round(results_true['ANGLE_SUM'],2):>2}\n        ARW: {np.round(rim_area_disc_area,2):>6.2f}"
        i += 1
        plt.subplot(nrows, ncols, i)
        plt.imshow(display_rim_true)
        plt.axis('off')
        plt.title(title,**mono_font)

        # Model
        rim_area_disc_area_model = results['RIM_AREA'] / results['DISC_AREA']
        if results['TYPE'] == "Connected":
            title = f" Rim/Disc: {np.round(results['RIM_DISC_RATIO'],2):.2f}\n      ARW: {np.round(rim_area_disc_area,2):.2f}"
        else:
            title = f"Rim Angle: {np.round(results['ANGLE_SUM'],2):>2}\n        ARW: {np.round(rim_area_disc_area_model,2):>6.2f}"
        i += 1
        plt.subplot(nrows, ncols, i)
        plt.imshow(display_rim)
        plt.axis('off')
        plt.title(title,**mono_font)

        plt.tight_layout()
        fig.set_size_inches(8, 6)
        fig.savefig('..//images//20220426-062458-Rim-multiclass_best//rdr angle radar//'+results['IMAGE'],
                    bbox_inches='tight',
                    pad_inches=.1,
                    dpi=300)
        plt.close(fig)
        #########################

        #########################
        # Image Disc, Rim/Disc, Angle, DDLS
        nrows = 1
        ncols = 3
        i = 0
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols)

        # Raw
        i += 1
        plt.subplot(nrows, ncols, i)
        plt.imshow(cv2.imread(img_path)[:, :, ::-1])
        plt.axis('off')
        plt.title('')

        # Clinician
        disc_size = ""
        if results_true['DISC_SIZE'] == 1:
            disc_size = "Small Disc"
        elif results_true['DISC_SIZE'] == 2:
            disc_size = "Average Disc"
        elif results_true['DISC_SIZE'] == 3:
            disc_size = "Large Disc"

        title_center = ""
        if results_true['TYPE'] == "Connected":
            title_center = f"Rim/Disc: {np.round(results_true['RIM_DISC_RATIO'],2):.2f}"
        else:
            title_center = "Angle: " + str(np.round(results_true['ANGLE_SUM'],2))

        title = f"{disc_size:<16}{title_center:<17}DDLS: {results_true['DDLS_SCORE']:<2}"

        i += 1
        plt.subplot(nrows, ncols, i)
        plt.imshow(display_rim_true)
        plt.axis('off')
        plt.title(title, fontsize=8)

        # Model
        disc_size = ""
        if results['DISC_SIZE'] == 1:
            disc_size = "Small Disc"
        elif results['DISC_SIZE'] == 2:
            disc_size = "Average Disc"
        elif results['DISC_SIZE'] == 3:
            disc_size = "Large Disc"

        title_center = ""
        if results['TYPE'] == "Connected":
            title_center = f"Rim/Disc: {np.round(results['RIM_DISC_RATIO'],2):.2f}"
        else:
            title_center = "Angle: " + str(np.round(results['ANGLE_SUM'],2))

        title = f"{disc_size:<16}{title_center:<17}DDLS: {results['DDLS_SCORE']:<2}"

        i += 1
        plt.subplot(nrows, ncols, i)
        plt.imshow(display_rim)
        plt.axis('off')
        plt.title(title, fontsize=8)

        plt.tight_layout()
        fig.set_size_inches(8, 6)
        fig.savefig('..//images//20220426-062458-Rim-multiclass_best//ddls//'+results['GROUP']+"//"+results['IMAGE'],
                    bbox_inches='tight',
                    pad_inches=.1,
                    dpi=300)
        plt.close(fig)
        #########################
    #################

    #################
    # Save the results
    df_results = df_results_empty.copy()
    df_results = df_results.append(results, ignore_index=True)

    df_results.to_csv(results_save_path, mode='a', header=False, index=False)
    #################
