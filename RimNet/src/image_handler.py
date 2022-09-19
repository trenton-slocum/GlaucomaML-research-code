# -*- coding: utf-8 -*-
"""
Class will handle the images in terms of folders

Moves images to train, test, val folders as needed

Separates images *DM.jpg and *CM.jpg
"""

import fnmatch
import glob
import os
import random
from pathlib import Path
from shutil import copyfile

import cv2
import numpy as np
import pandas as pd
import PIL
from matplotlib import pyplot as plt


def mask_seperator(folder_source, file_types, folder_dest_cm, folder_dest_dm):

    folder_dest = [folder_dest_cm, folder_dest_dm]
    for i, m in enumerate(('CM', 'DM')):
        #############################
        # Get the mask types
        files = []
        for f in file_types:
            files.extend(
                fnmatch.filter(os.listdir(folder_source), '*' + m + '.' + f))
        #############################

        #############################
        # Copy to destination
        if not os.path.exists(folder_dest[i]):
            os.makedirs(folder_dest[i])

        for f_i, f in enumerate(files):
            print(str(f_i + 1), len(files), f, sep=" : ")
            img_path = os.path.join(folder_source, f)
            save_path = os.path.join(folder_dest[i], f)
            copyfile(img_path, save_path)
        #############################

    # For missing files
    copied_files = fnmatch.filter(os.listdir(folder_dest_cm), '*')
    copied_files.extend(fnmatch.filter(os.listdir(folder_dest_dm), '*'))
    source_files = fnmatch.filter(os.listdir(folder_source), '*')

    print("Files not copied:", (set(source_files).difference(copied_files)))

    #
    #pd.DataFrame(fnmatch.filter(os.listdir(folder_dest_cm), '*')).to_csv('files//mask_list.csv')
    pd.DataFrame(fnmatch.filter(os.listdir(folder_dest_cm),
                                '*')).to_csv('files//mask_list_cm.csv')
    pd.DataFrame(fnmatch.filter(os.listdir(folder_dest_dm),
                                '*')).to_csv('files//mask_list_dm.csv')


def mask_seperator_refuge(folder_source,
                          folder_dest_cm,
                          folder_dest_dm,
                          invert=False):
    '''
    Use opencv color detection mask technique to get the cup and disc out of the image
    
    Assuming masks are .bmp files
    Assuming the mask / cup are in the same file. This funciton specifically for the
    Ref
    
    Parameters
    ----------
    folder_source : str
        Where the masks are
    folder_dest_cm : str
        Where to save the cup masks
    folder_dest_dm : str
        Where to save the disc masks
    invert : bool
        False : default cv mask, which is white mask against black background
        True  : invert the mask, black mask against white background 

    Returns
    -------
    None.

    '''
    files = fnmatch.filter(os.listdir(folder_source), '*.bmp')
    folder_dest = [folder_dest_cm, folder_dest_dm]

    for f, file in enumerate(files):
        print(str(f + 1), len(files), file, sep=' : ')

        for i, m in enumerate(('CM', 'DM')):
            lower_range = np.array([0, 0, 0])
            upper_range = np.array([10, 10, 10])

            if m == 'DM':
                lower_range = np.array([0, 0, 0])
                upper_range = np.array([160, 160, 160])

            img_path = os.path.join(folder_source, file)
            img = cv2.imread(img_path)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            mask = cv2.inRange(hsv, lower_range, upper_range)
            if invert:
                mask = 255 - mask
            '''
            plt.imshow(mask, cmap='gray')
            plt.show()
        
            plt.imshow(img, cmap='gray')
            plt.show()
            '''
            cv2.imwrite(os.path.join(folder_dest[i], file[:-3] + "jpg"), mask)


class ImageHandler():
    #def __init__(self):

    def image_count(self, directory):
        '''
        Returns
        -------
        Count of all the images and their labels (based on folder)
        '''
        results = dict()
        total = 0
        for folder in ('train', 'val', 'test'):
            count = len(
                glob.glob(
                    str(Path(directory + '/' + folder + '/raw/' + '/*.jpg'))))
            results[folder] = count
            total += count
        print(f"{'Total':>15}: {total:>6}")
        for folder, count in results.items():
            if folder == "TOTAL":
                next
            print(f"{folder:>15}: {count:>6} {round((count/total)*100,2):>6}%")

        results['TOTAL'] = total
        print("\n")

    def copy_images(self,
                    directory_raw_images,
                    directory_masks,
                    img_list,
                    destination,
                    image_type,
                    mask_type="",
                    resize=None):
        '''
        Parameters
        ----------
         directory_raw_images : string
            Directory of where the raw images exists
        directory_masks : string
            Directory of where the images of masks exists
        img_list : list
            list of image names without folder location or image type
        destination : string
            The root directory where to move the images
            Assumes two sub-directories within the root exist:
                mask
                raw
        image_type : string
            suffix of image type "jpg" "tiff" etc
        mask_type : string, optional
            string to add to mask if needed
            for example if we're masking CAR and BUILDING from same image, we can
            name the images:
                0001CAR.jpg
                0001BUILDING.jpg
            And we specify which mask with the mask_type
        resize : tuple, optional
            resize the image to this dimension

        Returns
        -------
        None.
        
        Test Files
        -------
        directory_raw_images = 'images/samples/images'
        directory_masks = 'images/samples/masks'
        destination = 'images/samples_split/train'
        mask_type = "DM"
        image_type = "jpg"
        resize=None
        '''
        for i, img in enumerate(img_list):
            print(i, len(img_list), img, sep=" : ")
            for folder in ['raw', 'mask']:
                img_path = ""
                save_path = ""

                if folder == "raw":
                    img_path = Path(directory_raw_images + "/" + img + "." +
                                    image_type)
                elif folder == "mask":
                    img_path = Path(directory_masks + "/" + img + mask_type +
                                    "." + image_type)

                save_path = Path(destination + "/" + folder + "/" + img + "." +
                                 image_type)

                if resize is None:
                    copyfile(img_path, save_path)
                else:
                    img_temp = PIL.Image.open(img_path).resize(resize)
                    img_temp.save(save_path)

    def copy_image_and_masks(self,
                             directory_raw_images,
                             directory_masks,
                             destination,
                             split,
                             image_type,
                             mask_type="",
                             resize=None):
        '''
        Parameters
        ----------
        directory_raw_images : string
            Directory of where the raw images exists
        directory_masks : string
            Directory of where the images of masks exists
            The function will make sure the raw image and the mask exist
            If both images don't exist, the image is not copied
            Assuming masks have the suffect CM.jpg or DM.jpg
        destination : string
            The directory where to move the images
            Creates directory if it does not exist
            Within this destination directory the function creates directories
            destination:
                train:
                    masks
                    raw
                val:
                    masks
                    raw
                test:
                    masks
                    raw
            The split argument determines how many images go in each
        split : list or tuple, 3x1
            the train, validation, test split percentages
            Value can be zero and that particular group will get none
            must add up to 1 or function returns with error
        mask_type : string
            string to add to mask if needed
            for example if we're masking CAR and BUILDING from same image, we can
            name the images:
                0001CAR.jpg
                0001BUILDING.jpg
            And we specify which mask with the mask_type
        image_type : string
            suffix of the image types "jpg" for example
        resize : tuple, optional
            resize the image to this dimension

        Returns
        -------
        images : list
            First element is a count of the images moved
            Second element is a list of all the raw images not moved due to not finding mask
            Second element is a list of all the masks not moved due to not finding raw image
            
        Test Files
        -------
        directory_raw_images = 'images/samples/images'
        directory_masks = 'images/samples/masks'
        destination = 'images/samples_split/'
        split = (.8, .1, .1)
        mask_type = "DM"
        image_type = "jpg"
        '''
        #######################
        # Check if the splits add up to 1, should be in percentages
        if sum(split) != 1:
            print("Split percentages exceed 100%")
            return

        raw_images = glob.glob(
            str(Path(directory_raw_images + '/*.' + image_type)))
        raw_masks = glob.glob(
            str(Path(directory_masks + '/*' + mask_type + '.' + image_type)))
        #######################

        #######################
        # Images with both masks and raw
        r = set(
            [Path(x).name[:(-1 * len(image_type) - 1)] for x in raw_images])
        m = set([
            Path(x).name[:(-1 * len(image_type) - 1 - len(mask_type))]
            for x in raw_masks
        ])
        raw_images_masks = list(r.intersection(m))
        raw_images_no_masks = r.difference(m)
        masks_no_raw_images = m.difference(r)

        if len(raw_images_masks) == 0:
            print("No matching masks and raw images.")
            return
        random.shuffle(raw_images_masks)
        #######################

        #######################
        # Split the masks
        # How many total images do we have?
        # How many of each type?
        data_size = len(raw_images_masks)
        train_size = int(split[0] * data_size)
        val_size = int(split[1] * data_size)
        test_size = data_size - train_size - val_size

        print(f"{'Dataset size':>20}: {data_size:>4}")
        print(f"{'Training size':>20}: {train_size:>4}")
        print(f"{'Validation size':>20}: {val_size:>4}")
        print(f"{'Test size':>20}: {test_size:>4}")
        #######################

        #######################
        # Make the directory if it doesn't eixt
        destination = str(Path(destination))
        Path(destination).mkdir(parents=True, exist_ok=True)

        for i in ['mask', 'raw']:
            Path(destination + "/train/" + i).mkdir(parents=True,
                                                    exist_ok=True)
            Path(destination + "/val/" + i).mkdir(parents=True, exist_ok=True)
            Path(destination + "/test/" + i).mkdir(parents=True, exist_ok=True)
        #######################

        #######################
        # Move images
        print("Copying training images.")
        img_list = raw_images_masks[:train_size]
        self.copy_images(directory_raw_images, directory_masks, img_list,
                         destination + "/train", image_type, mask_type, resize)

        print("Copying validation images.")
        img_list = raw_images_masks[train_size:(train_size + val_size)]
        self.copy_images(directory_raw_images, directory_masks, img_list,
                         destination + "/val", image_type, mask_type, resize)

        print("Copying test images.")
        img_list = raw_images_masks[(train_size + val_size):]
        self.copy_images(directory_raw_images, directory_masks, img_list,
                         destination + "/test", image_type, mask_type, resize)
        #######################

        return ([raw_images_masks, raw_images_no_masks, masks_no_raw_images])


if __name__ == "__main__":
    folder_sources = [
        'images//REFUGE//Annotation-Training400//Disc_Cup_Masks//Glaucoma',
        'images//REFUGE//Annotation-Training400//Disc_Cup_Masks//Non-Glaucoma',
        'images//REFUGE//REFUGE-Validation400-GT//Disc_Cup_Masks',
        'images//REFUGE//REFUGE-Test-GT//Disc_Cup_Masks//G',
        'images//REFUGE//REFUGE-Test-GT//Disc_Cup_Masks//N'
    ]

    for i, folder in enumerate(folder_sources):
        print(str(i + 1), len(folder_sources), folder, sep=' : ')
        folder_dest_dm = 'images//REFUGE_All_DM'
        folder_dest_cm = 'images//REFUGE_All_CM'
        mask_seperator_refuge(folder,
                              folder_dest_cm,
                              folder_dest_dm,
                              invert=False)

        folder_dest_dm = 'images//REFUGE_All_DM_INVERTED'
        folder_dest_cm = 'images//REFUGE_All_CM_INVERTED'
        mask_seperator_refuge(folder,
                              folder_dest_cm,
                              folder_dest_dm,
                              invert=True)
