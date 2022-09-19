# -*- coding: utf-8 -*-
"""
Functions for calculating DDLS
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

        # Draw the line to check for intersection
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
