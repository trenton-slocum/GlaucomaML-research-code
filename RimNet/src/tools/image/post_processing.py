# -*- coding: utf-8 -*-
"""
Post processing used throughout our model building process.

Some of these functions are no longer used for our final model
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import splev, splprep


##############################
def find_area(img):
    '''
    img : 2D gray scale unit8
    '''
    contours_img, hierarchy = cv.findContours(img, cv.RETR_LIST,
                                              cv.CHAIN_APPROX_SIMPLE)

    if False:
        temp = np.zeros_like(img, np.uint8)
        cv.drawContours(temp, contours_img, -1, (255, 255, 255), 10)
        plt.imshow(temp, cmap='gray')
        plt.imshow(img, cmap='gray')

    return (cv.contourArea(contours_img[0]))


##############################


##############################
# Find largest contour
def find_largest_contour(result_mask):
    '''
    Parameters
    ----------
    result_mask : numpy.ndarray
        Assuming it is the mask result from the model, which is already in
        2D gray scale unit8, so we don't need to apply the threshold

    Returns
    -------    
    mask : 2D gray scale unit8

    '''
    # Find the contours, keep largest
    (contours, _) = cv.findContours(
        result_mask.astype(np.uint8).copy(), cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE)

    # No contours found
    if len(contours) == 0: return result_mask

    largest_contour = max(contours, key=cv.contourArea)

    # Approximate the contour
    epsilon = 0.004 * cv.arcLength(largest_contour, True)
    largest_contour = cv.approxPolyDP(largest_contour, epsilon, True)

    # Create the new mask
    mask = np.ones(result_mask.shape[:], dtype="uint8") * 255
    mask = 255 - cv.drawContours(mask, [largest_contour], 0, 0, -1)

    mask = 1 - cv.threshold(mask, 128, 1, cv.THRESH_BINARY_INV)[1]
    mask = mask.astype('float32')
    return mask


##############################


##############################
# Apply morphology
# Opening and closing
def apply_morp(img, plot=False):
    '''
    Apply opening then closing to get rid of tiny noise
    Maybe even pinch off some noise
    Get structuring element/kernel
    which will be used for closing operation
    
    Parameters
    ----------
    img : numpy.ndarray, 2D gray scale unit8

    Returns
    -------    
    img_moprhed : 2D gray scale unit8

    '''

    closingSize = 1

    # Selecting an elliptical kernel
    element = cv.getStructuringElement(
        cv.MORPH_ELLIPSE, (2 * closingSize + 1, 2 * closingSize + 1),
        (closingSize, closingSize))

    img_moprhed = img
    img_moprhed = cv.morphologyEx(img_moprhed,
                                  cv.MORPH_OPEN,
                                  element,
                                  iterations=2)
    img_moprhed = cv.morphologyEx(img_moprhed,
                                  cv.MORPH_CLOSE,
                                  element,
                                  iterations=2)

    if plot:
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(img_moprhed, cmap='gray')
        plt.axis('off')

    return (img_moprhed)


##############################


##############################
def find_largest_connected_component(img,
                                     dilate=False,
                                     erode=False,
                                     plot=False):
    '''
    Find the largest connected component
    
    Parameters
    ----------
    img : numpy.ndarray, 2D gray scale unit8
    dilate : bool, if True we apply dilation
    erode : bool, if True we apply erosion
    plot : bool, if True we plot the results

    Returns
    -------    
    img_cca : 2D gray scale unit8

    '''

    kernal = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    th, img_cca = cv.threshold(img, 50, 255, cv.THRESH_BINARY)

    if dilate:
        img_cca = cv.dilate(img_cca, kernal)

    # Connected components with stats.
    nb_components, output, stats, centroids = cv.connectedComponentsWithStats(
        img_cca, connectivity=4)

    # Find the largest non background component.
    # Note: range() starts from 1 since 0 is the background label.
    max_label, max_size = max([(i, stats[i, cv.CC_STAT_AREA])
                               for i in range(1, nb_components)],
                              key=lambda x: x[1])
    img_cca = 255 * (output == max_label).astype('uint8')

    if erode:
        img_cca = cv.erode(img_cca, kernal)

    if plot:
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(img_cca, cmap='gray')
        plt.axis('off')

    return (img_cca)


##############################


##############################
def apply_convex_hull(img, plot=False):
    '''
    Convex Hull
    The smallest or the tight-fitting convex boundary
    is known as a convex hull.
    
    Parameters
    ----------
    img : numpy.ndarray, 2D gray scale unit8


    Returns
    -------    
    img_convex_hull : 2D gray scale unit8

    '''

    img_convex_hull = np.zeros_like(img)
    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL,
                                          cv.CHAIN_APPROX_NONE)
    hull = cv.convexHull(contours[0])
    cv.fillPoly(img_convex_hull, [hull], color=(255))
    #cv2.drawContours(img_convex_hull, contours, -1, (0), 1)

    # Find the cup
    contours, hierarchy = cv.findContours(img, cv.RETR_TREE,
                                          cv.CHAIN_APPROX_NONE)
    if len(contours) != 2:
        return img

    cup = contours[1]
    if cv.contourArea(contours[1]) > cv.contourArea(contours[0]):
        cup = contours[0]
    cv.fillPoly(img_convex_hull, [cup], color=(0))
    cv.drawContours(img_convex_hull, [cup], -1, (255), 0)

    if plot:
        plt.subplot(1, 3, 1)
        plt.imshow(img, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(img_convex_hull, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(np.abs(img - img_convex_hull), cmap='gray')
        #plt.imshow(cv2.bitwise_and(img, img_convex_hull), cmap='gray')
        plt.axis('off')

    return (img_convex_hull)


##############################


##############################
def apply_smoothing(img, plot=False):
    '''
    Apply Smoothing
    The smallest or the tight-fitting convex boundary
    is known as a convex hull.
    
    Parameters
    ----------
    img : numpy.ndarray, 2D gray scale unit8


    Returns
    -------    
    img_convex_hull : 2D gray scale unit8

    '''

    # Let's consider just the disc
    # cv2.CHAIN_APPROX_NONE parameter instructs findContours() to return all boundry points
    # cv2.RETR_EXTERNAL parameter instructs findContours() to return only outer most contours, in our case the disc
    contours, hierarchy = cv.findContours(img, cv.RETR_TREE,
                                          cv.CHAIN_APPROX_NONE)

    if len(contours) != 2:
        return img

    smoothened = []
    # scipy.interpolate.Rb
    for contour in contours:
        x, y = contour.T
        # Convert from numpy arrays to normal arrays
        x = x.tolist()[0]
        y = y.tolist()[0]

        # Find the B-spline representation of an N-D curve.
        tck, u = splprep([x, y], s=len(x))
        # Evaluate a B-spline or its derivatives
        x_new, y_new = splev(u, tck, der=0)

        # Convert it back to numpy format for opencv to be able to display it
        res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new, y_new)]
        smoothened.append(np.asarray(res_array, dtype=np.int32))

    # We smoothed our contours
    # Create two masks and apply bitwise operations
    cont_1 = np.zeros_like(img)
    cv.fillPoly(cont_1, pts=[smoothened[0]], color=(255))

    cont_2 = np.zeros_like(img)
    cv.fillPoly(cont_2, pts=[smoothened[1]], color=(255))

    # Invert the smaller area

    if (cv.contourArea(smoothened[0]) < cv.contourArea(smoothened[1])):
        cont_1 = cv.bitwise_not(cont_1)
    else:
        cont_2 = cv.bitwise_not(cont_2)

    img_smoothed = cv.bitwise_and(cont_1, cont_2)

    if plot:
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(img_smoothed, cmap='gray')
        plt.title("")
        plt.axis('off')

    return (img_smoothed)


##############################

##############################


def extract_disc_cup(img):
    disc, cup = None, None
    # Find the disc, cup
    contours, hierarchy = cv.findContours(img, cv.RETR_TREE,
                                          cv.CHAIN_APPROX_NONE)

    contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)

    disc = contours[0]
    cup = contours[1]

    if cv.contourArea(cup) > cv.contourArea(disc):
        disc = contours[1]
        cup = contours[0]

    # Find center of either
    M = cv.moments(cup)
    x_center = int(round(M["m10"] / M["m00"]))
    y_center = int(round(M["m01"] / M["m00"]))

    # Fill up from the center
    disc_img = img.copy()
    cv.floodFill(disc_img, None, (x_center, y_center), 255)

    # Cup is the bitwise
    cup_img = cv.bitwise_and(disc_img, cv.bitwise_not(img))

    if False:
        plt.imshow(disc_img, cmap='gray')
        plt.imshow(cup_img, cmap='gray')
        plt.imshow(img, cmap='gray')

        cv.countNonZero(disc_img)
        cv.countNonZero(cup_img)
        cv.countNonZero(img)
        cv.countNonZero(disc_img) - cv.countNonZero(cup_img)
        cv.countNonZero(disc_img) - cv.countNonZero(cup_img) - cv.countNonZero(
            img)

    return (disc_img, cup_img)


def extract_disc_cup_contours(img):
    disc, cup = None, None

    # Find the cup
    contours, hierarchy = cv.findContours(img, cv.RETR_TREE,
                                          cv.CHAIN_APPROX_NONE)
    disc = contours[0]
    cup = contours[1]

    if cv.contourArea(cup) > cv.contourArea(cup):
        disc = contours[1]
        cup = contours[0]

    disc_img = np.zeros_like(img)
    cv.fillPoly(disc_img, pts=[disc], color=(255))

    #cup_img = np.zeros_like(img)
    #cv.fillPoly(cup_img, pts = [cup], color=(255))

    cup_img = cv.bitwise_and(disc_img, cv.bitwise_not(img))
    cup_img = find_largest_connected_component(cup_img)

    if False:
        plt.imshow(disc_img, cmap='gray')
        plt.imshow(cup_img, cmap='gray')
        plt.imshow(img, cmap='gray')

    return (disc_img, cup_img)


##############################
