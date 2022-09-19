# -*- coding: utf-8 -*-
"""
Applies the model to the images and display the results in a 2x4 image.
Also generates CSV describing performance.
raw image : true disk : true cup : true rim
          : pred disk : pred cup : pred rim

"""

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

print(tf.__version__)
import glob
import os
import pathlib
import random
import time
from collections import defaultdict

import PIL
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

print("Current Working Directory" + os.getcwd())
import fnmatch

random.seed(10)

import segmentation_models as sm
import tools.calculators.DDLS_Calculator as ddls
import tools.image.post_processing as pp

###########################


def dice(y_true, y_pred):
    """Calculate dice similarity coefficient given [W, H] input
    Arguments:
        y_true: [w, h] numeric array where each value is 0 or 1
        y_pred: [w, h] numeric array where each value is 0 or 1
    Returns:
        Dice similarity coefficient for this particular pair.
    """
    y_true = np.equal(y_true, 1)
    y_pred = np.equal(y_pred, 1)

    union = np.logical_and(y_pred, y_true).sum()

    score = (2 * union) / (y_true.sum() + y_pred.sum())
    return score


def iou_coef(y_true, y_pred):
    iou = np.zeros(y_true.shape[0])
    #total_pixels = 2 * y_true.shape[1] * y_true.shape[2] * y_true.shape[3]
    total_pixels = 2 * y_true.shape[0] * y_true.shape[1]
    for k in range(0, y_true.shape):
        print()
        difference = y_true[k] - y_pred[k]
        intersection = np.count_nonzero(difference == 0)
        image_iou = intersection / (total_pixels - intersection)
        iou[k] = image_iou
    meaniou = np.average(iou)
    return meaniou


def iou_helper(result1, result2, num_classes=3):
    '''Intersection Over Union for each class

    Parameters
    ----------
    result1 : 2d numpy.ndarray, with integer items in range 0 ... num_classes -1
    result2 : 2d numpy.ndarray, with integer items in range 0 ... num_classes -1

    Returns
    -------
    list
        Calcaulted IoU for each class

    '''
    iou_list = []
    for value in range(num_classes):
        filtered_res1 = (result1 == value)
        filtered_res2 = (result2 == value)

        intersection = np.logical_and(filtered_res1, filtered_res2)
        union = np.logical_or(filtered_res1, filtered_res2)
        iou_score = np.sum(intersection) / np.sum(union)
        iou_list.append(iou_score)

    return iou_list


def iou(result1, result2, num_classes=3):
    '''
    Mean Intersection Over Union for each class

    Parameters
    ----------
    result1 : numpy.ndarray
    result2 : numpy.ndarray

    Returns
    -------
    float
        Calcaulted IoU

    '''

    iou_list = iou_helper(result1, result2, num_classes)

    return np.mean(iou_list)


def is_connected(predicted_rim, error_checking=False):
    '''Check if a predicted [W,H] rim with 0/255 values is connected or not

    Returns True if connected, False otherwise
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

    '''

    predicted_rim_disconnected_test = pp.find_largest_connected_component(
        predicted_rim, dilate=False, erode=False)
    if error_checking:
        plt.imshow(predicted_rim_disconnected_test, cmap='gray')

    _, hierarchy = cv2.findContours(predicted_rim_disconnected_test,
                                    cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    if hierarchy.shape[1] >= 2:
        return True
    elif hierarchy.shape[1] == 1:
        return False
    else:
        raise RuntimeError('Error in hierarchy')


def calc_areas(mask):
    """Calculate areas of rim, disc, and cup for a [w, h] array with integer values."""
    # Disc = rim + cup, or any nonzero portion
    disc_area = np.sum(np.greater(mask, 0))
    rim_area = np.sum(np.equal(mask, 1))
    cup_area = np.sum(np.equal(mask, 2))

    return (rim_area, disc_area, cup_area)


def calc_rdar(mask):
    """Calculate rim to disk area ratio for [W, H] array."""
    rim_area, disc_area, _ = calc_areas(mask)

    return rim_area / disc_area


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


# TODO(Tyler): Factor this out, as it is lifted from the data pipeline
def apply_clahe(image):
    # Assuming color is in RGB since read in by tf.image.decode_jpeg

    img_hist = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_hist[:, :, 0] = clahe.apply(img_hist[:, :, 0])

    # convert the YUV image back to RGB format
    image = cv2.cvtColor(img_hist, cv2.COLOR_YUV2RGB)

    return image


def diameters_summary(df_diameters):
    min_rims = df_diameters[df_diameters['rim/disc'] == np.min(
        df_diameters['rim/disc'])].head(1)
    temp = {
        'mean_rim': np.mean(df_diameters['rim_diam']),
        'min_ratio': min_rims['rim/disc'].values[0],
        'min_disc': min_rims['disc_diam'].values[0],
        'min_cup': min_rims['cup_diam'].values[0],
        'min_rim': min_rims['rim_diam'].values[0]
    }
    return (temp)


###########################


class MyMeanIOU(tf.keras.metrics.MeanIoU):

    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(y_true, tf.argmax(y_pred, axis=-1),
                                    sample_weight)


#cup_model = load_model("models/20210307-132558-transfer-cm", custom_objects={'MyMeanIOU':MyMeanIOU})
#disk_model = load_model("models/20210306-121316-transfer-dm", custom_objects={'MyMeanIOU':MyMeanIOU})
# rim_model = load_model("models/20210316-233421-transfer-rm", custom_objects={'MyMeanIOU':MyMeanIOU})
# rim_model = load_model("models/20210411-164520-transfer-rm-dropout", custom_objects={'MyMeanIOU':MyMeanIOU})
# rim_model = load_model(
#     "../models/20210901-162158-rm-big-shuffle-with-flips-224_best",
#     custom_objects={'MyMeanIOU': MyMeanIOU})
# "../models/20211012-155715-rm-new-backbone_tuned_mp_best"
rim_model = load_model("../models/20220426-062458-Rim-multiclass_best",
                       custom_objects={'iou_score': sm.metrics.iou_score})

print(rim_model.summary())
tf.keras.utils.plot_model(rim_model,
                          expand_nested=True,
                          to_file='./model_plot.png',
                          show_shapes=True)


def create_mask(pred_mask):
    """ Converts a W,H,N mask to W,H with integer labels"""
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def calc_accuracy(true_mask, pred_mask):
    """Calculates accuracy of a pair of [W, H] inputs."""
    is_correct = np.equal(true_mask, pred_mask)
    total_correct = is_correct.sum()
    fraction_correct = total_correct / true_mask.size
    return fraction_correct


########################
#load models
# disk_model = load_model('Disk Model - Trial 9 - 89% Acc, 89% Acc.h5')
#disk_model.summary()
#disk_model.get_weights()
#disk_model.optimizer
# cup_model = load_model('Cup Mask - Trial 6 - 95% Acc, 96% Val Acc.h5')
# Dimensions of the image
_, img_width, img_height, img_channels = rim_model.layers[0].input_shape[0]
########################

###########################
# Locations to our images

# Use these to switch between datasets
DRISHTI_L1_SPLIT_PATH = '../images/Drishti_Level_1_split//'
INTERNAL_PATH = '../images/4.24.22 Broken-Whole Rim Ready'

gpu_machine = True
if (gpu_machine):
    raw_image_path = INTERNAL_PATH
    rim_image_path = INTERNAL_PATH
    save_image_path = '../images/multiclass_4.24.22_outputs'
    # save_image_path = '../images/comparison_resized_iou_rim_split_oct_tuned_224'
else:
    raw_image_path = DRISHTI_L1_SPLIT_PATH
    rim_image_path = DRISHTI_L1_SPLIT_PATH
    cup_image_path = DRISHTI_L1_SPLIT_PATH
    disk_image_path = DRISHTI_L1_SPLIT_PATH
    save_image_path = '../images/drishti_oct_tuned_224_split'

# Create durectories for saving
if not os.path.exists(save_image_path):
    os.makedirs(save_image_path)

for suffix in ['train', 'val', 'test']:
    curr_path = save_image_path + '//' + suffix
    if not os.path.exists(curr_path):
        os.makedirs(curr_path)

###########################

###########################
# Save results

CSV_NAME = '..//files_output//4.25.22_multiclass_results_header.csv'
df_results = pd.DataFrame(
    columns=('IMAGE', 'TYPE', 'RIM_IOU_BG', 'RIM_IOU_RIM', 'RIM_IOU_CUP',
             'RIM_IOU_MEAN', 'RIM_ACCURACY', 'TRUE_DISC_DIAM', 'TRUE_CUP_DIAM',
             'TRUE_RIM_DIAM', 'TRUE_RATIO', 'TRUE_MEAN_RIM',
             'PRED_PP_DISC_DIAM', 'PRED_PP_CUP_DIAM', 'PRED_PP_RIM_DIAM',
             'PRED_PP_RATIO', 'PRED_PP_MEAN_RIM', 'RIM_AREA_TRUE',
             'DISC_AREA_TRUE', 'CUP_AREA_TRUE', 'RDAR_TRUE', 'RIM_AREA_MODEL',
             'DISC_AREA_MODEL', 'CUP_AREA_MODEL', 'RDAR_MODEL', 'ANGLE_TRUE',
             'ANGLE_PRED'))
df_results_empty = df_results
df_results.to_csv(CSV_NAME, mode='w', header=True, index=False)
###########################


def print_stats(iou_scores, dice_scores, accuracy_scores):
    """Calculates and prints aggregate metrics given per-image scores."""
    iou_per_class = np.array(iou_scores).mean(axis=0)

    print(f'Mean Accuracy: {np.mean(accuracy_scores)}')
    print(f'Background IoU: {iou_per_class[0]}')
    print(f'Rim IoU: {iou_per_class[1]}')
    print(f'Cup IoU: {iou_per_class[2]}')
    print(f'Mean IoU {np.mean(iou_per_class)}')
    # print(f'Dice mean: {np.mean(dice_scores)}')


def gen_figure(display_list, filename, split):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        if i == 2 or i == 1:
            plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        else:
            plt.imshow(display_list[i])
        plt.axis('off')
    plt.savefig(os.path.join(save_image_path, split, filename))
    plt.close()


def find_angle_wrapper(predicted_rim, predicted_disc, img_raw):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
        predicted_rim, connectivity=4)
    # Ignore the first element since 0 is always going to be the largest, it is the background
    largest_components = np.argsort(stats[:, cv2.CC_STAT_AREA])[:-1][::-1]

    # If we have multiple disconnects, limit to the top 2
    if len(largest_components) > 2:
        largest_components = largest_components[0:2]

    if np.max(predicted_rim) < 255:
        # Generate the disconnected rim
        predicted_rim = 255 * np.isin(output,
                                      largest_components).astype('uint8')

    return ddls.find_angle(predicted_rim, predicted_disc, img_raw)


def image_to_mask(image):
    """Converts an image to a mask.
        Args:
            image: A black, white, and blue image.
        Returns:
            an image where black maps to 0, white to 1, and blue to 2
    """
    image = np.asarray(image)

    # If there's a transparency layer, ditch it
    if image.shape[-1] == 4:
        print(f'shape is : {image.shape}')
        image = image[:, :, :3]
        print(f'shape is fixed to: {image.shape}')

    pixel_sums = np.sum(image, axis=-1)

    rim_mask = np.equal(pixel_sums, 255 * 3).astype(np.uint8)
    cup_mask = np.equal(pixel_sums, 255).astype(np.uint8) * 2

    # Mask will be 0, 1, and 2 exclusively
    final_mask = rim_mask + cup_mask
    return final_mask


#for i, img_path in enumerate(raw_images[:10]):
error_checking = False

splits = ['val', 'train']

# No test split for drishti
if 'drishti' not in raw_image_path.lower():
    splits.append('test')

for split in splits:
    all_ious = []
    all_dice = []
    all_accuracy = []

    # Stats for performance after post processing
    all_ious_pp = []
    all_dice_pp = []
    all_accuracy_pp = []

    raw_images = glob.glob(
        str(pathlib.Path(f'{raw_image_path}//{split}//raw//*')))
    true_masks = glob.glob(
        str(pathlib.Path(f'{rim_image_path}//{split}//mask//*')))

    print(f'Split: {split}, total images: {len(raw_images)}')
    for i, img_path in enumerate(raw_images):
        start = time.time()
        if img_path.endswith(".png"):
            print(str(i + 1),
                  len(raw_images),
                  os.path.basename(img_path),
                  sep=' : ')
            ###########################
            # Find true cup / disk & load it, if not found skip
            # Also resize it to the size used for prediction.
            img_path_mask = fnmatch.filter(true_masks,
                                           "*" + os.path.basename(img_path))

            img_mask = PIL.Image.open(img_path_mask[0]).resize(
                (img_width, img_height), PIL.Image.NEAREST)
            # TODO(tyler): new rim area calculation
            # rim_area = np.clip(np.asarray(img_rim), 0, 1)

            ###########################

            ###########################
            # Predict
            # Predicted mask is 0/1
            img = load_image(img_path, img_width, img_height, show=False)
            raw_img = img
            img = apply_clahe(np.squeeze(img.astype(np.uint8)))
            img = np.expand_dims(img, 0).astype(np.float32)

            true_mask = image_to_mask(img_mask)

            # output is w, h, 3
            predicted_rim = rim_model.predict(img)

            if predicted_rim.shape[-1] == 1:
                # This output only has one class in it.
                # We need to convert back to 2 output layers for the
                # rest of the processing
                new_rim = np.ones(predicted_rim.shape) - predicted_rim

                predicted_rim = np.concatenate([new_rim, predicted_rim],
                                               axis=-1)

            # Should be w, h, 1
            predicted_mask = create_mask(predicted_rim)

            print('figure generation')
            gen_figure([
                np.squeeze(raw_img) / 255,
                np.expand_dims(true_mask, -1), predicted_mask
            ], os.path.basename(img_path), split)

            # TODO(Tyler): Get largest component before calculating areas
            # and IoUs
            ious = iou_helper(true_mask, np.squeeze(predicted_mask))
            print(ious)
            all_ious.append(ious)
            accuracy = calc_accuracy(true_mask, np.squeeze(predicted_mask))
            all_accuracy.append(accuracy)

            true_rdar = calc_rdar(true_mask)
            pred_rdar = calc_rdar(np.squeeze(predicted_mask))

            pred_rim_area, pred_disc_area, pred_cup_area = calc_areas(
                np.squeeze(predicted_mask))
            true_rim_area, true_disc_area, true_cup_area = calc_areas(
                true_mask)

            # Now we calculate RDAR at the thinnest point, or the missing angle
            # if incomplete. We need to repeat for truth and pred

            # First we resize to get the right aspect ratio
            # TODO(tyler): Why the -1?
            img_raw = cv2.imread(img_path)[:, :, ::-1]
            resize_width = img_raw.shape[1]
            resize_height = img_raw.shape[0]

            img_raw = np.asarray(img_raw)

            pred_mask_resized = cv2.resize(np.squeeze(predicted_mask).astype(
                np.uint8), (resize_width, resize_height),
                                           interpolation=cv2.INTER_NEAREST)
            true_mask_resized = cv2.resize(true_mask.astype(np.uint8),
                                           (resize_width, resize_height),
                                           interpolation=cv2.INTER_NEAREST)

            # The tools below expect the mask to be all white, or 255
            pred_mask_sums = pred_mask_resized
            # We need to clean off any extra blobs by finding largest connected component
            pred_rim_mask = pp.find_largest_connected_component(
                np.equal(pred_mask_sums, 1).astype(np.uint8) * 255,
                dilate=False,
                erode=False)
            pred_disc_mask = pp.find_largest_connected_component(
                np.greater(pred_mask_sums, 0).astype(np.uint8) * 255,
                dilate=False,
                erode=False)

            true_mask_sums = true_mask_resized
            true_rim_mask = np.equal(true_mask_sums, 1).astype(np.uint8) * 255
            true_disc_mask = np.greater(true_mask_sums, 0).astype(
                np.uint8) * 255

            # TODO(tyler): Make this loop into a stateless function
            # Clear variables from last loop
            true_degrees = None
            pred_degrees = None
            df_diameters_pred = defaultdict(lambda: None)
            df_diameters = defaultdict(lambda: None)

            # check predicted
            if is_connected(pred_rim_mask):
                # Calculate RDAR
                display_rim_predicted_pp, df_diameters_all = ddls.find_diameters(
                    np.squeeze(img_raw),
                    pred_rim_mask,
                    line_width=10,
                    circle_diam=25,
                    cup_thickness=10,
                    disc_thickness=10,
                    rim_thickness=15)

                df_diameters_pred = diameters_summary(df_diameters_all)
            else:
                # Find the missing angle
                print('angles')
                display_img, angle_counts = find_angle_wrapper(
                    pred_rim_mask, pred_disc_mask, np.squeeze(img_raw))
                pred_degrees = np.sum(angle_counts)

            # Check true
            if is_connected(true_rim_mask):
                # Calculate RDAR
                display_rim_predicted_pp, df_diameters_all = ddls.find_diameters(
                    np.squeeze(img_raw),
                    true_rim_mask,
                    line_width=10,
                    circle_diam=25,
                    cup_thickness=10,
                    disc_thickness=10,
                    rim_thickness=15)

                df_diameters = diameters_summary(df_diameters_all)
            else:
                # Find the missing angle
                print('angles')
                display_img, angle_counts = find_angle_wrapper(
                    true_rim_mask, true_disc_mask, np.squeeze(img_raw))
                true_degrees = np.sum(angle_counts)

            #         ###########################
            #         # Save iou results
            df_results = df_results.append(
                {
                    'IMAGE': os.path.basename(img_path),
                    'TYPE': split,
                    'RIM_IOU_BG': ious[0],
                    'RIM_IOU_RIM': ious[1],
                    'RIM_IOU_CUP': ious[2],
                    'RIM_IOU_MEAN': np.mean(ious),
                    'RIM_ACCURACY': accuracy,
                    # 'RIM_DICE': rim_dice,
                    # 'RIM_DICE_P': rim_dice_processed,
                    'TRUE_DISC_DIAM': df_diameters['min_disc'],
                    'TRUE_CUP_DIAM': df_diameters['min_cup'],
                    'TRUE_RIM_DIAM': df_diameters['min_rim'],
                    'TRUE_RATIO': df_diameters['min_ratio'],
                    'TRUE_MEAN_RIM': df_diameters['mean_rim'],
                    'PRED_PP_DISC_DIAM': df_diameters_pred['min_disc'],
                    'PRED_PP_CUP_DIAM': df_diameters_pred['min_cup'],
                    'PRED_PP_RIM_DIAM': df_diameters_pred['min_rim'],
                    'PRED_PP_RATIO': df_diameters_pred['min_ratio'],
                    'PRED_PP_MEAN_RIM': df_diameters_pred['mean_rim'],
                    'RIM_AREA_TRUE': true_rim_area,
                    'DISC_AREA_TRUE': true_disc_area,
                    'CUP_AREA_TRUE': true_cup_area,
                    'RDAR_TRUE': true_rdar,
                    'RIM_AREA_MODEL': pred_rim_area,
                    'DISC_AREA_MODEL': pred_disc_area,
                    'CUP_AREA_MODEL': pred_cup_area,
                    'RDAR_MODEL': pred_rdar,
                    'ANGLE_TRUE': true_degrees,
                    'ANGLE_PRED': pred_degrees
                },
                ignore_index=True)
            #         ###########################

            #         ###########################
            # Write file
            df_results.to_csv(CSV_NAME, mode='a', header=False, index=False)
            df_results = df_results_empty.copy()
    #         ###########################

    #         ###########################
    #         # Save model predictions
    #         if True:
    #             cv2.imwrite(
    #                 os.path.join(save_image_path + f'//{split}_pred//',
    #                              pathlib.Path(img_path).stem) + ".jpg",
    #                 predicted_rim_processed_resized)
    #         ###########################

    #         ###########################
    #         # Show results
    #         # raw fundus image
    #         nrows = 2
    #         ncols = 4
    #         i = 0
    #         fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
    #         i += 1
    #         plt.subplot(nrows, ncols, i)
    #         plt.imshow(img_raw)
    #         plt.axis('off')
    #         #plt.show()

    #         # Rim Smaller
    #         i += 1
    #         plt.subplot(nrows, ncols, i)
    #         plt.imshow(rim_area, cmap='gray')
    #         plt.axis('off')
    #         plt.title("Rim True Down")

    #         # Rim
    #         i += 1
    #         plt.subplot(nrows, ncols, i)
    #         plt.imshow(rim_area_raw, cmap='gray')
    #         plt.axis('off')
    #         plt.title("Rim True")
    #         #plt.show()

    #         # Display Narrowest Rim
    #         i += 1
    #         plt.subplot(nrows, ncols, i)
    #         plt.imshow(display_rim)
    #         plt.axis('off')
    #         plt.title(f"{np.round(df_diameters['min_ratio'],2):.2f}")
    #         #plt.show()

    #         #### 2nd row
    #         # Raw-resized
    #         i += 1
    #         plt.subplot(nrows, ncols, i)
    #         plt.imshow(img[0].astype(np.uint8))
    #         plt.title('Image Used')

    #         # Predicted rim
    #         i += 1
    #         plt.subplot(nrows, ncols, i)
    #         plt.imshow(predicted_rim_processed, cmap='gray')
    #         plt.axis('off')
    #         plt.title(
    #             f"{rim_iou_processed[0]:.2f}, {rim_iou_processed[1]:.2f}, {np.mean(rim_iou_processed):.2f}"
    #         )

    #         # Predicted rim resized
    #         i += 1
    #         plt.subplot(nrows, ncols, i)
    #         plt.imshow(predicted_rim_processed_resized, cmap='gray')
    #         plt.axis('off')
    #         plt.title(
    #             f"{rim_iou_processed_resized[0]:.2f}, {rim_iou_processed_resized[1]:.2f}, {np.mean(rim_iou_processed_resized):.2f}"
    #         )

    #         # Narrowest
    #         i += 1
    #         plt.subplot(nrows, ncols, i)
    #         plt.imshow(display_rim_predicted_pp.astype(np.uint8))
    #         plt.axis('off')
    #         plt.title(
    #             f"{np.round(df_diameters_predicted_pp['min_ratio'],2):.2f}")
    #         #################################

    #         plt.tight_layout()
    #         fig.set_size_inches(8, 6)
    #         fig.savefig(os.path.join(save_image_path + f'//{split}//',
    #                                  pathlib.Path(img_path).stem) + ".jpg",
    #                     dpi=300)
    #         plt.close(fig)
    #         ###########################
    print(f'Stats for non-post processed {split}')
    print_stats(all_ious, all_dice, all_accuracy)

    # print('\n')
    # print(f'Stats after post processing {split}')
    # print_stats(all_ious_pp, all_dice_pp, all_accuracy_pp)
    # print('\n')
