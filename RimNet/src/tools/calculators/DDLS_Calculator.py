# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 12:16:25 2020

@author: Haroon

Calculate DDLS


Good explanation on the bounding box on contours
https://theailearner.com/tag/cv2-minarearect/
"""

import math
import time
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tools.image.post_processing as pp

warnings.filterwarnings("error")


def point_on_circle(center, radius, angle):
    '''
        Finding the x,y coordinates on circle, based on given angle
    '''
    angle = math.radians(angle)
    x = center[0] + (radius * math.cos(angle))
    y = center[1] + (radius * math.sin(angle))

    return (int(x), int(y))


def find_angle(predicted_rim, predicted_disc, raw_image):
    '''
    Input:
        predicted_rim : 2D black & white array of the rim
            used to calculate the angle

        predicted_disc : 2D black & white array of the disc from DiscModel
            used to find the center of the disc

        raw_image : 3D color image of the fundus photos to overlay the results on


    Returns:
        display_img:
            3D array of the raw_image with the predicted rim & angle overlayed
            Center of disc is colored in white
            Predicted rim is outlined in white
            Angle start is in green and angle end is in red

        angle_counts:
            1D array of the angles found
            Clinically, at most this should be one. Function does not check for this
    '''
    #########
    # Will use the passed disc to find the center to calcualte the angle from
    contours, hierarchy = cv2.findContours(predicted_disc, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)

    M = cv2.moments(contours[0])
    x_center = int(round(M["m10"] / M["m00"]))
    y_center = int(round(M["m01"] / M["m00"]))
    center = np.array((x_center, y_center))

    if False:
        display_img = predicted_rim.copy()
        cv2.circle(display_img, (x_center, y_center), 25, (255, 255, 255), -1)
        plt.imshow(display_img, cmap='gray')

    #########

    #########
    # We need to draw lines from the center to some end point
    # The end point must pass the rim mask so we can check if pixels are overlapping
    # We'll make the minimum length of the line the length of the top right
    # to the bottom left of the iamge
    top_left = (0, 0)
    bot_right = (predicted_rim.shape[1], predicted_rim.shape[0])
    line_length = int(
        np.round(np.linalg.norm(np.array(top_left) - np.array(bot_right))))
    #########

    #########
    # Calculate all the end points from the center to the line_length for 360 degrees
    angles = np.radians(np.arange(0, 361, 1, dtype=int))
    x_points = (center[0] + (line_length * np.cos(angles))).astype(int)
    y_points = (center[1] + (line_length * np.sin(angles))).astype(int)

    if False:
        display_img = raw_image.copy()

        point = (x_points[0], y_points[0])
        cv2.line(display_img, center, point, (255, 0, 0), 10)

        point = (x_points[90], y_points[90])
        cv2.line(display_img, center, point, (0, 255, 0), 10)

        point = (x_points[180], y_points[180])
        cv2.line(display_img, center, point, (0, 0, 255), 10)

        point = (x_points[270], y_points[270])
        cv2.line(display_img, center, point, (255, 0, 255), 10)

        cv2.circle(display_img, (x_center, y_center), 25, (255, 255, 255), -1)
        plt.imshow(display_img)
    #########

    ############
    # Find a starting point
    # We want a starting point to be intersecting
    angle_off_set = 0
    blank_image = np.zeros_like(predicted_rim)
    for angle in range(0, 361, 1):
        cv2.line(
            blank_image, center,
            point_on_circle(center=center, radius=line_length, angle=angle),
            (255, 255, 255), 3)
        if np.logical_and(predicted_rim, blank_image).any():
            angle_off_set = angle
            break

        cv2.line(
            blank_image, center,
            point_on_circle(center=center, radius=line_length, angle=angle),
            (0, 0, 0), 3)

    if False:
        contours, hierarchy = cv2.findContours(predicted_rim,
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_NONE)
        display_img = raw_image.copy()
        cv2.circle(display_img, center, 25, (255, 255, 255), -1)
        cv2.drawContours(display_img, contours, -1, (255, 255, 0), 10)
        cv2.line(
            display_img, center,
            point_on_circle(center=center,
                            radius=line_length,
                            angle=angle_off_set), (0, 0, 255), 5)
        plt.imshow(display_img)
    ############

    ############
    '''
    We move 360 degrees around drawing lines from the center to the edge
    of the image. We start from the center to the first point of the rim contour
    It does not matter which point of the rim contour we start,
    as long as we start at an intersection.

    We move one degree each step looking for the first non-intersection, that is
    when the line we draw from the center to the edge of the image does NOT touch the rim
    This marks the beginning of our angle.
    We continue to step until we find the next intersection, this marks the end of our angle

    Approach to calculaing the angle:
    Count the steps between the first non-intersect (open angle) to the first intersect (close angle)
    Originally I saved the three points and calculating the angle, but this did not work for angles
    > 180 degrees
    '''
    display_img = raw_image.copy()
    contours_mask, hierarchy = cv2.findContours(predicted_rim,
                                                cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(display_img, contours_mask, -1, (255, 255, 255), 10)
    cv2.circle(display_img, center, 40, (255, 255, 255), -1)

    on = False
    prev_intersect = True
    points_angles = []
    angle_pair = []
    prev_point = None
    t = 0

    angle_counts = []
    angle_count = 0

    blank_image = np.zeros_like(predicted_rim)

    import time
    program_starts = time.time()
    for angle in range(0, 361, 1):
        #print(theta)

        # Get the point of the next degree
        point = point_on_circle(center=center,
                                radius=line_length,
                                angle=angle + angle_off_set)

        # For error checking
        if angle == 0:
            pass
            #cv2.line(display_img, center, point, (0, 0, 255), 5)

        # Draw it on a blank canvass
        #blank_image = np.zeros_like(predicted_rim)
        cv2.line(blank_image, center, point, (255, 255, 255), 3)

        if False:
            cv2.drawContours(blank_image, contours_mask, -1, (255, 255, 255),
                             2)
            plt.imshow(blank_image, cmap='gray')

        # Check for intersection
        this_intersects = np.logical_and(predicted_rim, blank_image).any()
        if not this_intersects:
            # If we're still looking for the next non-intersect, that is our start
            # Otherwise, we increase our angle count, meaning we've still not found the next
            # intersect, where our angle ends
            if not on:
                cv2.line(display_img, center, point, (0, 255, 0), 10)
                angle_pair.append(point)
                on = True
                angle_count = 1
            else:
                angle_count += 1
                #break
        elif not prev_intersect and on:
            cv2.line(display_img, center, prev_point, (255, 0, 0), 10)
            angle_pair.append(prev_point)
            points_angles.append(angle_pair)

            angle_pair = []

            # Grab the angle count and reset counter
            angle_counts.append(angle_count)
            angle_count = 0

            # Reset, look for next no-intersect
            on = False
            t += 1

        #if t == 2:
        #break
        #cv2.line(display_img, center, point, (0, 255, 0), 2)
        prev_intersect = this_intersects
        prev_point = point

        # Remove line
        cv2.line(blank_image, center, point, (0, 0, 0), 3)

    if False:
        plt.imshow(display_img)
    print("It has been {0} seconds since the loop started".format(
        time.time() - program_starts))
    ############

    return (display_img, angle_counts)


def find_angle_testing():
    import glob
    import os
    import pathlib

    import segmentation_models as sm
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image

    disc_model = load_model(
        "models/20220131-230633-dm-new-backbone_tuned_mp_best",
        custom_objects={'iou_score': sm.metrics.iou_score})
    _, img_width, img_height, disc_img_channels = disc_model.layers[
        0].input_shape[0]

    path_masks = "..//images//broken_rim_tests//mask_scaled//"
    path_raw = "..//images//broken_rim_tests//raw//"
    raw_images = glob.glob(str(pathlib.Path(f'{path_raw}//*.jpg')))

    df_results_empty = pd.DataFrame(columns=('IMAGE', 'ANGLES', 'ANGLE_SUM'))
    df_results_empty.to_csv(
        '..//files_output//Disconnected_Rim_Angles__2022_03_08.csv',
        mode='w',
        header=False,
        index=False)

    for i, img_path in enumerate(raw_images):
        print((i + 1), len(raw_images), img_path, sep=" : ")
        #img_path = '..\\images\\broken_rim_tests\\raw\\AXISDEV_48144_115678_20150203132907307f8fe9f0f326e5fb9_3.jpg'

        #########
        # Mask
        mask = path_masks + pathlib.Path(img_path).stem + ".MASK.jpg"
        if not os.path.exists(mask):
            continue
        #break
        mask = cv2.imread(mask)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)[1]
        #########

        #########
        # Disc Mask
        img = image.load_img(img_path, target_size=(img_width, img_height))
        img_tensor = image.img_to_array(img)  # (height, width, channels)
        img_tensor = np.expand_dims(img_tensor, axis=0)

        if False:
            plt.imshow(np.squeeze(img_tensor) / 255)
            plt.axis('off')
            plt.show()

        img = img_tensor
        img = apply_clahe(np.squeeze(img.astype(np.uint8)))
        img = np.expand_dims(img, 0).astype(np.float32)
        predicted_disc = disc_model.predict(img)

        if predicted_disc.shape[-1] == 1:
            # This output only has one class in it.
            # We need to convert back to 2 output layers for the
            # rest of the processing
            new_disc = np.ones(predicted_disc.shape) - predicted_disc

            predicted_disc = np.concatenate([new_disc, predicted_disc],
                                            axis=-1)

        predicted_disc = create_mask(predicted_disc)
        predicted_disc = 255 * predicted_disc.astype(np.uint8)
        predicted_disc = pp.find_largest_connected_component(predicted_disc,
                                                             dilate=False,
                                                             erode=False)

        # Convert image back to original dimensions
        img_raw = cv2.imread(img_path)[:, :, ::-1]
        resize_width = img_raw.shape[1]
        resize_height = img_raw.shape[0]
        predicted_disc = cv2.resize(predicted_disc,
                                    (resize_width, resize_height),
                                    interpolation=cv2.INTER_LINEAR)

        if False:
            plt.imshow(predicted_disc, cmap='gray')
        #########

        #########
        if False:
            nrows = 2
            ncols = 1
            i = 0
            fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
            i += 1
            plt.subplot(nrows, ncols, i)
            plt.imshow(mask, cmap='gray')
            plt.axis('off')
            plt.title(" ")

            # Rim Smaller
            i += 1
            plt.subplot(nrows, ncols, i)
            plt.imshow(predicted_disc, cmap='gray')
            plt.axis('off')
            plt.title(" ")

        # For outlining the segmentaiton
        contours, hierarchy = cv2.findContours(predicted_disc,
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_NONE)
        display_img_rim_disc = np.ascontiguousarray(
            cv2.imread(img_path)[:, :, ::-1], dtype=np.uint8)
        display_img_disc = np.ascontiguousarray(
            cv2.imread(img_path)[:, :, ::-1], dtype=np.uint8)
        display_img_rim = np.ascontiguousarray(
            cv2.imread(img_path)[:, :, ::-1], dtype=np.uint8)
        cv2.drawContours(display_img_rim_disc, contours, -1, (255, 255, 0), 10)
        cv2.drawContours(display_img_disc, contours, -1, (255, 255, 255), 10)

        contours_mask, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(display_img_rim_disc, contours_mask, -1,
                         (255, 255, 255), 10)
        cv2.drawContours(display_img_rim, contours_mask, -1, (255, 255, 255),
                         10)

        # Center
        M = cv2.moments(contours[0])
        x_center = int(round(M["m10"] / M["m00"]))
        y_center = int(round(M["m01"] / M["m00"]))
        cv2.circle(display_img_rim_disc, (x_center, y_center), 25,
                   (255, 255, 255), -1)

        if False:
            plt.imshow(display_img_rim_disc)
            plt.imshow(display_img_disc)
            plt.imshow(display_img_rim)
        #########

        #########
        '''
        contours_mask, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Center
        M = cv2.moments(contours_mask[0])
        x_center = int(round(M["m10"]/M["m00"]))
        y_center = int(round(M["m01"]/M["m00"]))

        if pathlib.Path(img_path).stem == '1.3.6.1.4.1.29565.1.4.0.18030.1453957176.77818':
            x_center = 380
            y_center = 200
        '''
        center = np.array((x_center, y_center))

        # Long enough line
        bot_left = (0, mask.shape[0])
        top_left = (0, 0)
        bot_right = (mask.shape[1], mask.shape[0])
        top_right = (mask.shape[1], 0)

        if False:
            display_img = np.ascontiguousarray(
                cv2.imread(path_masks + os.path.basename(img_path)[:-4] +
                           ".MASK.jpg")[:, :, ::-1],
                dtype=np.uint8)

            cv2.circle(display_img, center, 20, (255, 255, 255), -1)
            cv2.circle(display_img, bot_left, 20, (0, 255, 0), -1)
            cv2.circle(display_img, top_left, 20, (0, 255, 0), -1)
            cv2.circle(display_img, bot_right, 20, (255, 0, 0), -1)
            cv2.circle(display_img, top_right, 20, (255, 100, 0), -1)

            plt.imshow(display_img)

        # Find longest line to work with
        starting_point = [bot_left, top_left, bot_right, top_right]
        distances = np.array([
            np.linalg.norm(center - np.array(starting_point[0])),
            np.linalg.norm(center - np.array(starting_point[1])),
            np.linalg.norm(center - np.array(starting_point[2])),
            np.linalg.norm(center - np.array(starting_point[3]))
        ])
        line_length = int(np.round(np.max(distances), 0))
        ############

        ############
        # Find a starting point
        # We want a starting point to be intersecting
        angle_off_set = 0
        for angle in range(0, 361, 1):
            blank_image = np.zeros_like(mask)
            cv2.line(
                blank_image, center,
                point_on_circle(center=center, radius=line_length,
                                angle=angle), (255, 255, 255), 3)
            if np.logical_and(mask, blank_image).any():
                angle_off_set = angle
                break

        if False:
            display_img = display_img_rim.copy()

            cv2.circle(display_img, center, 25, (255, 255, 255), -1)
            cv2.line(
                display_img, center,
                point_on_circle(center=center,
                                radius=line_length,
                                angle=angle_off_set), (0, 0, 255), 5)
            plt.imshow(display_img)
        ############

        ############
        '''
        We move 360 degrees around drawing lines from the center to the edge
        of the image. We start from the center to the first point of the rim contour
        It does not matter which point of the rim contour we start,
        as long as we start at an intersection.

        We move one degree each step looking for the first non-intersection, that is
        when the line we draw from the center to the edge of the image does NOT touch the rim
        This marks the beginning of our angle.
        We continue to step until we find the next intersection, this marks the end of our angle

        Approach to calculaing the angle:
        Count the steps between the first non-intersect (open angle) to the first intersect (close angle)
        Originally I saved the three points and calculating the angle, but this did not work for angles
        > 180 degrees
        '''
        display_img = display_img_rim.copy()
        cv2.drawContours(display_img, contours_mask, -1, (255, 255, 255), 2)
        cv2.circle(display_img, center, 25, (255, 255, 255), -1)

        on = False
        prev_intersect = True
        points_angles = []
        angle_pair = []
        prev_point = None
        t = 0

        angle_counts = []
        angle_count = 0
        for angle in range(0, 361, 1):
            #print(theta)

            # Get the point of the next degree
            point = point_on_circle(center=center,
                                    radius=line_length,
                                    angle=angle + angle_off_set)

            # For error checking
            if angle == 0:
                cv2.line(display_img, center, point, (0, 0, 255), 5)

            # Draw it on a blank canvass
            blank_image = np.zeros_like(mask)
            cv2.line(blank_image, center, point, (255, 255, 255), 3)

            if False:
                cv2.drawContours(blank_image, contours_mask, -1,
                                 (255, 255, 255), 2)
                plt.imshow(blank_image, cmap='gray')

            # Check for intersection
            if not np.logical_and(mask, blank_image).any():
                # If we're still looking for the next non-intersect, that is our start
                # Otherwise, we increase our angle count, meaning we've still not found the next
                # intersect, where our angle ends
                if not on:
                    cv2.line(display_img, center, point, (0, 255, 0), 5)
                    angle_pair.append(point)
                    on = True
                    angle_count = 1
                else:
                    angle_count += 1
                    #break
            elif not prev_intersect and on:
                cv2.line(display_img, center, prev_point, (255, 0, 0), 5)
                angle_pair.append(prev_point)
                points_angles.append(angle_pair)

                angle_pair = []

                # Grab the angle count and reset counter
                angle_counts.append(angle_count)
                angle_count = 0

                # Reset, look for next no-intersect
                on = False
                t += 1

            #if t == 2:
            #break
            #cv2.line(display_img, center, point, (0, 255, 0), 2)
            prev_intersect = np.logical_and(mask, blank_image).any()
            prev_point = point

        # Save the results
        df_results = df_results_empty.copy()
        df_results = df_results.append(
            {
                'IMAGE': os.path.basename(img_path),
                'ANGLES': angle_counts,
                'ANGLES_SUM': np.sum(angle_counts)
            },
            ignore_index=True)

        df_results.to_csv(
            '..//files_output//Disconnected_Rim_Angles__2022_03_08.csv',
            mode='a',
            header=False,
            index=False)
        '''
        plt.imshow(display_img)
        plt.axis('off')
        plt.title(str(np.round(angles,1)) + " : " + str(np.round(np.sum(angles),1)))
        plt.savefig(os.path.join("..//images//broken_rim_tests//test//" ,
                                 pathlib.Path(img_path).stem)+".jpg",
                    bbox_inches='tight',
                    pad_inches = .1,
                    dpi=300)
        plt.close()
        '''
        if True:
            raw_img = np.ascontiguousarray(cv2.imread(img_path)[:, :, ::-1],
                                           dtype=np.uint8)
            plt.figure(figsize=(15, 5))
            # Placing the plots in the plane
            plot1 = plt.subplot2grid((1, 5), (0, 0))
            plot2 = plt.subplot2grid((1, 5), (0, 1))
            plot3 = plt.subplot2grid((1, 5), (0, 2))
            plot4 = plt.subplot2grid((1, 5), (0, 3))
            plot5 = plt.subplot2grid((1, 5), (0, 4))

            # Raw
            plot1.imshow(raw_img)
            plot1.axis('off')

            # Disc
            plot2.imshow(display_img_disc)
            plot2.axis('off')

            # Rim
            plot3.imshow(display_img_rim)
            plot3.axis('off')

            # Rim + Disc
            plot4.imshow(display_img_rim_disc)
            plot4.axis('off')

            # Raw + Rim + Angle
            plot5.imshow(display_img)
            plot5.axis('off')
            plot5.set_title(
                str(angle_counts) + " : " + str(np.sum(angle_counts)))
            #plt.show()

            plt.savefig(os.path.join("..//images//broken_rim_tests//test//",
                                     pathlib.Path(img_path).stem) + ".jpg",
                        bbox_inches='tight',
                        pad_inches=.1,
                        dpi=300)
            plt.close()
        #########


def min_max_points_optimized(a, b, c_points):
    '''
    Uses angle between segements to find the closest (smallest angle) and furthers (largest angle)
    points

    Solution: https://manivannan-ai.medium.com/find-the-angle-between-three-points-from-2d-using-python-348c513e2cd

    a : the current cup point
    b : the center of the cup
    c_points : either the cup or the disc contour points
    '''
    ba = a - b
    bc = c_points[:, ] - b

    cosine_angle = np.sum(np.tile(ba, (bc.shape[0], 1)) * bc[:, ], axis=1) / (
        np.linalg.norm(ba) * np.linalg.norm(bc, axis=1))
    # Sometimes we get a number very close to 1 such as 1.0000000000000002
    # Which will throw a RunTime Warning if we try to apply inverse cosine (arccos)
    # Let's just round it, a pixel off makes no difference
    angles = np.arccos(np.round(cosine_angle, 10))

    point_closest = c_points[np.argmin(angles), :]
    point_furthest = c_points[np.argmax(angles), :]

    return (point_closest, point_furthest)


def min_max_points(a, b, c_points):
    '''
    Uses angle between segements to find the closest (smallest angle) and furthers (largest angle)
    points

    Solution: https://manivannan-ai.medium.com/find-the-angle-between-three-points-from-2d-using-python-348c513e2cd

    a : the current cup point
    b : the center of the cup
    c_points : either the cup or the disc contour points
    '''
    c = []
    angle_max = 0
    angle_min = 0
    point_closest = []
    point_furthest = []
    for pd_i, disc_point in enumerate(c_points):
        c = np.array(disc_point)

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba,
                              bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))

        try:
            angle = np.degrees(np.arccos(cosine_angle))
        except:
            # Sometimes we get a number very close to 1 such as
            # 1.0000000000000002
            # Which will throw a RunTime Warning, we can just round it
            angle = np.degrees(np.arccos(np.round(cosine_angle, 8)))

        if pd_i == 0:
            angle_max = angle
            angle_min = angle
            point_closest = disc_point
            point_furthest = disc_point

        if angle_min > angle:
            angle_min = angle
            point_closest = disc_point

        if angle_max < angle:
            angle_max = angle
            point_furthest = disc_point

    return (point_closest, point_furthest)


#img, predicted_rim_processed = img[0].copy(), rim_area
#img = img_raw
#rim = predicted_rim_processed_resized
def find_diameters_optimized(img,
                             rim,
                             line_width=2,
                             circle_diam=20,
                             cup_thickness=5,
                             disc_thickness=5,
                             rim_thickness=5):
    error_checking = False

    ###########################
    # Find smallest rim area
    # Find bottom 95% rim area
    # Find top 95% rim area
    # Find mean and median rim area
    disc, cup = pp.extract_disc_cup(rim)

    if error_checking:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Disc')
        ax1.imshow(rim, cmap='gray')
        ax2.imshow(disc - cup, cmap='gray')
        np.array_equal(rim * 255, disc - cup)

        plt.imshow(disc, cmap='gray')
        plt.imshow(cup, cmap='gray')

        plt.imshow(cv2.bitwise_not(cv2.bitwise_not(disc) - cup), cmap='gray')
        plt.imshow(cv2.bitwise_not(disc) - cup, cmap='gray')
        plt.imshow(disc - cup, cmap='gray')

    ########################
    # Find the cup contour
    contours_cup, hierarchy = cv2.findContours(cup, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_NONE)

    # the disc contour
    contours_disc, hierarchy = cv2.findContours(disc, cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_NONE)
    ########################

    ########################
    # We will rotate around the center of the cup
    M = cv2.moments(contours_cup[0])
    x_center = int(round(M["m10"] / M["m00"]))
    y_center = int(round(M["m01"] / M["m00"]))
    df_diameters = []
    rim_min = 0
    ########################

    if error_checking:
        img_copy = img.copy()
        cv2.circle(img_copy, (x_center, y_center), circle_diam, (0, 0, 255),
                   -1)
        cv2.drawContours(img_copy, contours_cup, -1, (255, 255, 255),
                         line_width)
        cv2.drawContours(img_copy, contours_disc, -1, (255, 255, 255),
                         line_width)
        plt.imshow(img_copy)

    ########################
    # For each point in the cup contours, we need to find:
    #   rim diameter, cup diameter, disc diameter
    # We this by finding the closest and furthest disc and cup points
    # That way, we can simply calculate distance between two points
    #############
    # Find the diameter of the rim, disc, cup
    # We do this by using the angles created by the segments from the center to the
    # cup point and disc point
    # Rim diameter: minimum angle current cup point (a), center (b), disc point (c)
    # Cup diameter: maximum angle current cup point (a), center (b), cup point (c)
    # Disc diameter: maximum angle current cup point (a), center (b), disc point (c)
    # We get the end points of each to draw the lines for illustration and error checking
    #
    # Solution: https://manivannan-ai.medium.com/find-the-angle-between-three-points-from-2d-using-python-348c513e2cd
    #############
    a = np.squeeze(contours_cup[0])
    b = np.array([x_center, y_center])
    c_points = np.squeeze(contours_disc)

    min_max_points_optimized(a, b, c_points)


#img, predicted_rim_processed = img[0].copy(), rim_area
#img = cv2.imread(img_path)[:,:,::-1]
#rim = predicted_rim
def find_diameters(img,
                   rim,
                   line_width=2,
                   circle_diam=20,
                   cup_thickness=5,
                   disc_thickness=5,
                   rim_thickness=5):
    error_checking = False

    ###########################
    # Find smallest rim area
    # Find bottom 95% rim area
    # Find top 95% rim area
    # Find mean and median rim area
    disc, cup = pp.extract_disc_cup(rim)

    if error_checking:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Disc')
        ax1.imshow(rim, cmap='gray')
        ax2.imshow(disc - cup, cmap='gray')
        np.array_equal(rim * 255, disc - cup)

        plt.imshow(disc, cmap='gray')
        plt.imshow(cup, cmap='gray')

        plt.imshow(cv2.bitwise_not(cv2.bitwise_not(disc) - cup), cmap='gray')
        plt.imshow(cv2.bitwise_not(disc) - cup, cmap='gray')
        plt.imshow(disc - cup, cmap='gray')

    ########################
    # Find the cup contour
    contours_cup, hierarchy = cv2.findContours(cup, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_NONE)

    # the disc contour
    contours_disc, hierarchy = cv2.findContours(disc, cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_NONE)
    ########################

    ########################
    # We will rotate around the center of the cup
    M = cv2.moments(contours_cup[0])
    x_center = int(round(M["m10"] / M["m00"]))
    y_center = int(round(M["m01"] / M["m00"]))
    df_diameters = []
    rim_min = 0
    ########################

    if error_checking:
        img_copy = img.copy()
        cv2.circle(img_copy, (x_center, y_center), circle_diam, (0, 0, 255),
                   -1)
        cv2.drawContours(img_copy, contours_cup, -1, (255, 255, 255),
                         line_width)
        cv2.drawContours(img_copy, contours_disc, -1, (255, 255, 255),
                         line_width)
        plt.imshow(img_copy)

    for i, point in enumerate(contours_cup[0]):
        if error_checking:
            print(i)

        #######################
        # Find the diameter of the rim, disc, cup
        # We do this by using the angles created by the segments from the center to the
        # cup point and disc point
        # Rim diameter: minimum angle current cup point (a), center (b), disc point (c)
        # Cup diameter: maximum angle current cup point (a), center (b), cup point (c)
        # Disc diameter: maximum angle current cup point (a), center (b), disc point (c)
        # We get the end points of each to draw the lines for illustration and error checking
        #
        # Solution: https://manivannan-ai.medium.com/find-the-angle-between-three-points-from-2d-using-python-348c513e2cd
        #######################
        a = np.array(point[0])
        b = np.array([x_center, y_center])
        '''
        start = time.time()
        disc_point_closest, disc_point_furthest = min_max_points(a, b, np.squeeze(contours_disc))
        cup_point_closest, cup_point_furthest = min_max_points(a, b, np.squeeze(contours_cup))
        end = time.time()
        print(end - start)
        '''

        if error_checking:
            start = time.time()

        disc_point_closest, disc_point_furthest = min_max_points_optimized(
            a, b, np.squeeze(contours_disc))
        cup_point_closest, cup_point_furthest = min_max_points_optimized(
            a, b, np.squeeze(contours_cup))

        if error_checking:
            print(time.time() - start)
        #######################

        #######################
        if error_checking:
            temp = np.zeros_like(img, np.uint8)
            cv2.circle(temp, (x_center, y_center), circle_diam, (0, 0, 255),
                       -1)
            cv2.drawContours(temp, contours_cup, -1, (255, 255, 255),
                             line_width)
            cv2.drawContours(temp, contours_disc, -1, (255, 255, 255),
                             line_width)

            # Cup
            cv2.circle(temp, (point[0]), circle_diam, (255, 0, 0), -1)
            cv2.line(temp, (point[0]), (x_center, y_center), (255, 0, 0),
                     thickness=line_width)
            cv2.circle(temp, (cup_point_furthest), circle_diam, (255, 0, 0),
                       -1)
            cv2.line(temp, (point[0]), (cup_point_furthest), (255, 0, 0),
                     thickness=line_width)

            # Disc
            cv2.circle(temp, disc_point_closest, circle_diam, (0, 255, 0), -1)
            cv2.line(temp, (x_center, y_center),
                     disc_point_closest, (255, 255, 0),
                     thickness=line_width)
            cv2.circle(temp, disc_point_furthest, circle_diam, (0, 255, 0), -1)
            cv2.line(temp, (x_center, y_center),
                     disc_point_furthest, (255, 255, 0),
                     thickness=line_width)

            # Rim
            cv2.line(temp, (point[0]),
                     disc_point_closest, (255, 255, 255),
                     thickness=line_width)
            plt.imshow(temp)
        #######################

        #######################
        # We have all the points we need, we can find diameters
        # math.sqrt( ((disc_point_closest[0]-disc_point_furthest[0])**2)+((disc_point_closest[1]-disc_point_furthest[1])**2) )
        disc_diameter = np.linalg.norm(disc_point_closest -
                                       disc_point_furthest)
        cup_diameter = np.linalg.norm(point[0] - cup_point_furthest)
        rim_diameter = np.linalg.norm(point[0] - disc_point_closest)
        #######################

        ##############################
        # Save the diameter and points
        df_diameters.append({
            'index': i,
            'disc_diam': disc_diameter,
            'disc_x1': disc_point_closest[0],
            'disc_y1': disc_point_closest[1],
            'disc_x2': disc_point_furthest[0],
            'disc_y2': disc_point_furthest[1],
            'cup_diam': cup_diameter,
            'cup_x1': point[0][0],
            'cup_y1': point[0][1],
            'cup_x2': cup_point_furthest[0],
            'cup_y2': cup_point_furthest[1],
            'rim_diam': rim_diameter,
            'rim_x1': point[0][0],
            'rim_y1': point[0][1],
            'rim_x2': disc_point_closest[0],
            'rim_y2': disc_point_closest[1],
            'rim/disc': rim_diameter / disc_diameter
        })
        ##############################

        ##############################
        if i == 0:
            rim_min = rim_diameter

        if rim_min >= rim_diameter:
            rim_min = rim_diameter
            display_rim = img.copy()

            cv2.circle(display_rim, (x_center, y_center), circle_diam,
                       (0, 0, 255), -1)
            cv2.drawContours(display_rim, contours_cup, -1, (255, 255, 255),
                             cup_thickness)
            cv2.drawContours(display_rim, contours_disc, -1, (255, 255, 255),
                             disc_thickness)

            # Cup
            cv2.circle(display_rim, tuple(point[0]), circle_diam, (0, 255, 0),
                       -1)
            cv2.line(display_rim,
                     tuple(point[0]), (x_center, y_center), (255, 255, 255),
                     thickness=line_width)
            #cv2.circle(display_rim, (cup_point_furthest), circle_diam, (255,0,0), -1)
            cv2.line(display_rim,
                     tuple(point[0]),
                     tuple(cup_point_furthest), (255, 255, 255),
                     thickness=line_width)

            # Disc
            cv2.circle(display_rim, tuple(disc_point_closest), circle_diam,
                       (0, 255, 0), -1)
            cv2.line(display_rim, (x_center, y_center),
                     tuple(disc_point_closest), (255, 255, 255),
                     thickness=line_width)
            #cv2.circle(display_rim, disc_point_furthest, circle_diam, (0,255,0), -1)
            cv2.line(display_rim, (x_center, y_center),
                     tuple(disc_point_furthest), (255, 255, 255),
                     thickness=line_width)

            # Rim
            cv2.line(display_rim,
                     tuple(point[0]),
                     tuple(disc_point_closest), (0, 0, 255),
                     thickness=rim_thickness)

            if error_checking:
                plt.imshow(display_rim)
        ##############################

    return (display_rim, pd.DataFrame(df_diameters))
    #plt.imshow(display_rim.astype(np.uint8))


def ddls_score(disc_size, ratio, angle=0):
    '''
    disc_size:
        1 : small
        2 : medium
        3 : large

    ratio : float

    angle : int
    '''

    if disc_size == 1:
        if angle > 180:
            return 10
        elif angle >= 91:
            return 9
        elif angle >= 46:
            return 8
        elif angle > 0:
            return 7

        if ratio < .1:
            return 6
        elif ratio <= .19:
            return 5
        elif ratio <= .29:
            return 4
        elif ratio <= .39:
            return 3
        elif ratio <= .49:
            return 2
        elif ratio >= .5:
            return 1

    elif disc_size == 2:
        if angle > 270:
            return 10
        elif angle >= 181:
            return 9
        elif angle >= 91:
            return 8
        elif angle >= 46:
            return 7
        elif angle > 0:
            return 6

        if ratio < .1:
            return 5
        elif ratio <= .19:
            return 4
        elif ratio <= .29:
            return 3
        elif ratio <= .39:
            return 2
        elif ratio >= .4:
            return 1

    elif disc_size == 3:
        if angle > 270:
            return 9
        elif angle >= 181:
            return 8
        elif angle >= 91:
            return 7
        elif angle >= 46:
            return 6
        elif angle > 0:
            return 5

        if ratio < .1:
            return 4
        elif ratio <= .19:
            return 3
        elif ratio <= .29:
            return 2
        elif ratio >= .3:
            return 1


'''
def Diameter_Calculator(img_path, angle, length):
    test_img = cv2.imread(img_path, 0)
    ret, thresh = cv2.threshold(test_img, 127, 255, 0)
    M = cv2.moments(thresh)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    dimensions = test_img.shape
    blank_img = np.zeros((dimensions[0], dimensions[1]))

    x = int(length * math.cos(angle))
    y = int(length * math.sin(angle))
    line_img = cv2.line(blank_img, (cX, cY), (cX + x, cY + y), 255, 1)
    line_img = cv2.line(blank_img, (cX, cY), (cX - x, cY - y), 255, 1)
    intersect = line_img * thresh

    coordinates = cv2.findNonZero(intersect)
    maxValues = np.amax(coordinates, axis=0)
    minValues = np.amin(coordinates, axis=0)

    x1 = maxValues[0][0]
    y1 = maxValues[0][1]
    x2 = minValues[0][0]
    y2 = minValues[0][1]
    diameter = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return diameter


#x = Diameter_Calculator('images/output/Second Pass Good RE/1.3.6.1.4.1.29565.1.4.0.18030.1453912506.63787_result_gs_gs.jpg', 1.5, 100 )
#print(x)
###Calculates DDLS on each image and outputs CSV with image name, angle, rim-disc ratio###
def DDLS_Calculator(dataset_path, length, slices, eye,
                    main_folder):  #use eye to distinguish files

    #create CSV File#
    with open(main_folder + '\\' + eye + 'DDLSResults.csv', 'w',
              newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            'Image Name', 'Angle', 'Rim-Disc Linear Ratio',
            'Rim-Disc Area Ratio', 'Cup-Disc Area Ratio', 'Width', 'Stage'
        ])

    #calculate DDLS#
    for i, file_name in enumerate(os.listdir(dataset_path)):
        img_path = dataset_path + '\\' + file_name
        o_img = cv2.imread(img_path, 0)
        ret, thresh = cv2.threshold(o_img, 127, 255, 0)
        M = cv2.moments(thresh)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        #Find dimensions, radian intervals, and create blank image of the image
        dimensions = o_img.shape
        interval = 2 * math.pi / slices

        #Initialize empty array
        width_array = np.empty(slices)
        intersect = 0

        #Find coordinates and add to width array
        for i in range(
                0, slices
        ):  #goes through all slices of the image. Stores the width.
            x = cX + int(length * math.cos(interval * i))
            y = cY + int(length * math.sin(interval * i))
            blank_img = np.zeros((dimensions[0], dimensions[1]))
            line_img = cv2.line(blank_img, (cX, cY), (x, y), 1, 1)
            intersect = line_img * thresh
            intersect = cv2.findNonZero(intersect)
            width = np.size(intersect, 0)
            width_array[i] = width

        #Find the degree at which the rim is the thinnest
        min_index = np.argmin(width_array)
        min_degree = min_index * interval

        #calculate the Rim Disc Linear Ratio
        diameter = Diameter_Calculator(img_path, min_degree, length)
        min_width = width_array[min_index]
        rim_disc_ratio = min_width / diameter

        #Calculate DDLS Stage
        stage = DDLS_Stager(min_width, rim_disc_ratio)

        #Calculate Rim-Disc Area Ratio and Cup-Disc Area Ratio#
        rim_disc_area_ratio, cup_disc_area_ratio = Ratios_Calculator.Ratios_Calculator(
            img_path)

        #append to CSV#
        with open(main_folder + '\\' + eye + 'DDLSResults.csv',
                  'a+',
                  newline='') as csvappend:
            writer = csv.writer(csvappend)
            writer.writerow([
                file_name,
                str(360 - min_index * interval * 180 / math.pi),
                str(round(rim_disc_ratio, 3)), rim_disc_area_ratio,
                cup_disc_area_ratio, min_width, stage
            ])


def DDLS_Stager(width, rim_disc_ratio):
    stage = 'staging was not initiated'
    pixel_to_mmsqrd = 0.000225  #found from manually masking images and counting pixels via
    pixel_side_to_mm = 0.01502
    width_mm = width * pixel_side_to_mm
    if width_mm < 1.5:
        if rim_disc_ratio >= 0.5:
            stage = '0a'
        elif rim_disc_ratio >= 0.4:
            stage = '0b'
        elif rim_disc_ratio >= 0.3:
            stage = '1'
        elif rim_disc_ratio >= 0.2:
            stage = '2'
        elif rim_disc_ratio >= 0.1:
            stage = '3'
        else:
            stage = '4 or greater'
    elif width_mm < 2:
        if rim_disc_ratio >= 0.4:
            stage = '0a'
        elif rim_disc_ratio >= 0.3:
            stage = '0b'
        elif rim_disc_ratio >= 0.2:
            stage = '1'
        elif rim_disc_ratio >= 0.1:
            stage = '2'
        else:
            stage = '3 or greater'
    else:
        if rim_disc_ratio >= 0.3:
            stage = '0a'
        elif rim_disc_ratio >= 0.2:
            stage = '0b'
        elif rim_disc_ratio >= 0.1:
            stage = '1'
        else:
            stage = '2 or greater'
    return stage
'''
