# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 21:00:37 2021

@author: EJMorales

Post processing to apply to our predicted images:
    keep largest contour:
        Will get rid of noise that is mis-masked
    blurring:
        smoot out the edges of the contour

Finding largest contour:
    https://stackoverflow.com/questions/56589691/how-to-leave-only-the-largest-blob-in-an-image/56591448
"""

##############################
'''
Discs:
    UCLA_axis01_45793_104848_003_2015012612502829045515065165b1cf1
    
Cups:
    UCLA_axis01_45804_104867_003_20150126125616363546dc4a8e78bb498
    UCLA_axis01_45793_104848_003_2015012612502829045515065165b1cf1
    UCLA_AXIS01_45791_165649_2015041010384487236ab39e74032d059_3
    UCLA_AXIS01_45791_165649_201504101038466152a6cd82be02cd528_4
    UCLA_axis01_45789_104838_004_20150126124841598ae39f3455f5b494d
    
'''
##############################

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


def sample_largest_contour():
    import PIL

    # Need to convert to gray scale
    image = 255 - cv.imread(
        'images//model_cm//UCLA_axis01_45804_104867_003_20150126125616363546dc4a8e78bb498.jpg'
    )
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Apply threshold to convert to binary b&w
    thresh = cv.threshold(gray, 128, 255, cv.THRESH_BINARY_INV)[1]

    # Apply contours and find largest one
    (contours, _) = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,
                                    cv.CHAIN_APPROX_SIMPLE)
    print(f"Contours found: {len(contours)}")

    largest_contour = max(contours, key=cv.contourArea)

    # Create a mask
    mask = np.ones(image.shape[:2], dtype="uint8") * 255
    mask = 255 - cv.drawContours(mask, [largest_contour], 0, 0, -1)
    mask = 1 - cv.threshold(mask, 128, 1, cv.THRESH_BINARY_INV)[1]
    plt.imshow(mask, cmap='gray')
    plt.axis('off')

    im = PIL.Image.fromarray(mask.astype('uint8'))
    im.save('images//test_contour.jpg')


##############################


##############################
# Find largest contour
def find_largest_contour(result_mask):
    '''

    Parameters
    ----------
    result_mask : numpy.ndarray
        Assuming it is the mask result from the model, which is already in
        binary b&w, so we don't need to apply the threshold

    Returns
    -------
    None.
    
    result_mask = result_cup

    '''
    pass
    # Assuming the mask is already in binary, 0 or 1

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

    #plt.imshow(mask, cmap='gray')
    #plt.axis('off')
    mask = 1 - cv.threshold(mask, 128, 1, cv.THRESH_BINARY_INV)[1]
    mask = mask.astype('float32')
    return mask


##############################

##############################
# Apply morphology
# Opening and closing


def apply_morp(img, plot=False):
    # Apply opening then closing to get rid of tiny noise
    # Maybe even pinch off some noise
    # Get structuring element/kernel
    # which will be used for closing operation
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
# Find the largest connected component
def find_largest_connected_component(img,
                                     dilate=False,
                                     erode=False,
                                     plot=False):
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
    # Convex Hull
    # The smallest or the tight-fitting convex boundary
    # is known as a convex hull.
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

if __name__ == "__main__":
    import os

    import pandas as pd

    # Show a sample
    pred_source = "..//images//samples_pp//UCLA_1.3.6.1.4.1.29565.1.4.0.18030.1453965279.80355"
    pred_source = "..//images//samples_pp//UCLA_1.3.6.1.4.1.29565.1.4.0.27655.1453793980.440898"
    rim_pred = cv.imread(pred_source + "_Pred.jpg", cv.IMREAD_GRAYSCALE)
    #rim_true = cv.imread(pred_source + "_Rim.jpg")
    img_raw = cv.imread(pred_source + "_Raw.jpg")
    rim_pred_pp = rim_pred
    img_name = os.path.basename(pred_source)
    step = 0

    #cv.imwrite(os.path.join(os.path.dirname(pred_source), "pp", img_name+"_"+str(step)+".jpg"), rim_pred)
    '''
    Find Largest Connected Component
    We can assume the model's predicted rim will have the largest area
    This may be an issue for predictd rims that are slightly disconnected
    For this reason, we'll dilate the image first, find the largest CC, then erode
    Doing so, also has the effect of filling up black holes
    '''
    step += 1
    rim_pred_pp = find_largest_connected_component(rim_pred_pp,
                                                   dilate=True,
                                                   erode=True,
                                                   plot=False)
    cv.imwrite(
        os.path.join(os.path.dirname(pred_source),
                     img_name + "_" + str(step) + ".jpg"), rim_pred_pp)
    '''
    Morphology
    We apply Opening and Closing
    This will fill in holes, get rid of small white "drips", possibly pinch off peninsulas
    '''
    step += 1
    rim_pred_pp = apply_morp(rim_pred_pp, plot=False)
    cv.imwrite(
        os.path.join(os.path.dirname(pred_source),
                     img_name + "_" + str(step) + ".jpg"), rim_pred_pp)
    '''
    Find Largest CC
    After morpholoy, we may have pinched off small pieces
    '''
    step += 1
    rim_pred_pp = find_largest_connected_component(rim_pred_pp,
                                                   dilate=False,
                                                   erode=False,
                                                   plot=False)
    cv.imwrite(
        os.path.join(os.path.dirname(pred_source),
                     img_name + "_" + str(step) + ".jpg"), rim_pred_pp)
    '''
    Convex Hull
    At this point, we're assuming a single connected component 
    We find the convex hull of the outer contour only (the disc). A convex hull is
    the smallest convex set that contains the contour    
    '''
    step += 1
    rim_pred_pp = apply_convex_hull(rim_pred_pp, plot=False)
    cv.imwrite(
        os.path.join(os.path.dirname(pred_source),
                     img_name + "_" + str(step) + ".jpg"), rim_pred_pp)
    '''
    B Spline Smoothing
    Smoothing is applied to both the disc and the cup
    '''
    step += 1
    rim_pred_pp = apply_smoothing(rim_pred_pp, plot=False)
    cv.imwrite(
        os.path.join(os.path.dirname(pred_source),
                     img_name + "_" + str(step) + ".jpg"), rim_pred_pp)

    ####################################
    # Find thinnest part
    disc, cup = extract_disc_cup(rim_pred_pp)
    contours, hierarchy = cv.findContours(rim_pred_pp, cv.RETR_EXTERNAL,
                                          cv.CHAIN_APPROX_NONE)

    display = img_raw.copy()
    disc_cup, h = cv.findContours(rim_pred_pp, cv.RETR_TREE,
                                  cv.CHAIN_APPROX_NONE)
    cv.drawContours(display, disc_cup, -1, (255, 255, 255), 1)

    M = cv.moments(contours[0])
    x_center = int(round(M["m10"] / M["m00"]))
    y_center = int(round(M["m01"] / M["m00"]))
    diameters = np.array([])
    diam_min = 0
    for i, point in enumerate(contours[0]):
        thickness = np.zeros_like(rim_pred_pp)
        cv.line(thickness, (x_center, y_center), (point[0][0], point[0][1]),
                (255, 0, 0), 2)
        thickness = cv.bitwise_and(thickness, cv.bitwise_not(cup))

        # Fit a circle and get the radius (Thickness)
        line_only = cv.bitwise_and(thickness, cv.bitwise_not(cup))
        c, h = cv.findContours(line_only, cv.RETR_EXTERNAL,
                               cv.CHAIN_APPROX_SIMPLE)
        ((x, y), radius) = cv.minEnclosingCircle(c[0])
        diameters = np.append(diameters, radius * 2)

        # Display
        if i == 0:
            diam_min = radius * 2

        if diam_min >= radius * 2:
            diam_min = radius * 2
            display = img_raw.copy()
            cv.drawContours(display, disc_cup, -1, (255, 255, 255), 1)
            cv.drawContours(display, c, -1, (255, 255, 255), -1)
    step += 1
    cv.imwrite(
        os.path.join(os.path.dirname(pred_source),
                     img_name + "_" + str(step) + ".jpg"), display)

    ####################################
    if False:
        diameters = np.array([])
        display = np.zeros_like(rim_pred_pp)
        contours, hierarchy = cv.findContours(rim_pred_pp, cv.RETR_EXTERNAL,
                                              cv.CHAIN_APPROX_NONE)
        disc_cup, h = cv.findContours(rim_pred_pp, cv.RETR_TREE,
                                      cv.CHAIN_APPROX_NONE)
        cv.drawContours(display, disc_cup, -1, (255, 255, 255), 1)
        for i, point in enumerate(contours[0]):
            if i % 5 == 0:
                print(i, point, len(contours[0]))
                thickness = np.zeros_like(rim_pred_pp)
                cv.line(thickness, (x_center, y_center),
                        (point[0][0], point[0][1]), (255, 0, 0), 2)
                thickness = cv.bitwise_and(thickness, cv.bitwise_not(cup))

                # Fit a circle and get the radius (Thickness)
                line_only = cv.bitwise_and(thickness, cv.bitwise_not(cup))
                c, h = cv.findContours(line_only, cv.RETR_EXTERNAL,
                                       cv.CHAIN_APPROX_SIMPLE)
                ((x, y), radius) = cv.minEnclosingCircle(c[0])
                diameters = np.append(diameters, radius * 2)
                '''
                # Display purposes
                display_circle = np.zeros_like(rim_pred_pp)
                cv.line(display_circle,(x_center,y_center),(point[0][0], point[0][1]),(255,0,0),1)
                display_circle = cv.bitwise_and(display_circle, cv.bitwise_not(cup))
                cv.circle(display_circle, (int(x),int(y)), int(round(radius)), (125,125,125), 1) 
                plt.imshow(display_circle, cmap='gray')
                plt.axis('off')
                plt.savefig("..//images//thickness//circle_" + str(i).zfill(3) + ".png")  
                '''

                # Display purposes:
                cv.line(display, (x_center, y_center),
                        (point[0][0], point[0][1]), (255, 0, 0), 1)
                display = cv.bitwise_and(display, cv.bitwise_not(cup))
                cv.drawContours(display, disc_cup, -1, (255, 255, 255), 1)
                cv.imwrite(
                    "..//images//thickness//line_" + str(i).zfill(3) + ".png",
                    display)

        import glob

        import imageio
        '''
        images = glob.glob("..//images//thickness//circle*")
        images = [imageio.imread(i) for i in images]
        imageio.mimsave('..//images//samples_pp//' + img_name + '_Cirscle.gif', images,  duration=0.1)    
        '''

        images = glob.glob("..//images//thickness//line*")
        images = [imageio.imread(i) for i in images]
        imageio.mimsave('..//images//samples_pp//' + img_name + '_Lines.gif',
                        images,
                        duration=0.1)
    ####################################
