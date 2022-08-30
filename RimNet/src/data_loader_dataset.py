# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 09:19:28 2020

@author: EJMorales

Modified version of dataloader from:
    https://github.com/HasnainRaz/SemSegPipeline

A TensorFlow Dataset API based loader for semantic segmentation problems.
"""

import random

import cv2
import numpy as np
import tensorflow as tf


class DataLoaderDataset():

    def __init__(self,
                 image_paths,
                 mask_paths,
                 image_size,
                 channels,
                 preprocessing=None):
        '''
        Parameters
        ----------
        image_paths : list of the image paths
        mask_paths : list of the mask paths
        image_size : (WIDTH, HEIGHT) tuple
        channels : int, number of channels in the image (usually 3)
        preprocessing: A function that accepts Image, Mask and processes them
        Returns
        -------
        None.

        '''

        self._image_paths = image_paths
        self._mask_paths = mask_paths
        self._image_size = image_size
        self._channels = channels
        # Min amount of image to retain in a crop
        self._crop_percent = .7
        # Max amount to shift an image, e.g. 20% of pixels shifted
        self._shift_fraction = .1
        self._random_generator = tf.random.Generator.from_seed(42)
        self.preprocessing = preprocessing

    def _gen_random_bool(self, prob_true=1):
        """Returns random bools, with fraction prob_true as True"""
        alpha = 1.0 - prob_true
        # For floats, the default range is [0, 1)
        rand_val = self._random_generator.uniform(shape=[1])
        return rand_val > alpha

    def _parse_data(self, image_paths, mask_paths):
        """
        Reads image and mask files depending on
        specified extension.

        https://www.tensorflow.org/api_docs/python/tf/io/decode_jpeg

        tf.io.decode_jpeg
        Decode a JPEG-encoded image to a uint8 tensor.
        """
        image_content = tf.io.read_file(image_paths)
        mask_content = tf.io.read_file(mask_paths)

        images = tf.image.decode_image(image_content,
                                       channels=self._channels[1],
                                       expand_animations=False)

        # Our masks are for 3 channels.
        # Rim: 255, 255, 255
        # Cup: all blue, so maybe 0, 0, 255
        # Background: All 0
        # We will make background class 0, rim class 1, cup class 2

        masks_raw = tf.image.decode_image(mask_content,
                                          channels=self._channels[1],
                                          expand_animations=False)

        # Resizing here could cause issues for masks as it could introduce
        # Greys, but yolo. nearest neighbor may help
        images, masks_raw = self._resize_data(images, masks_raw)

        # Sum the values in each channel
        mask_sums = tf.math.reduce_sum(masks_raw, axis=-1)
        # Background is all 0
        masks_background = tf.math.equal(mask_sums, 0)
        # Rim rim is white, so 255 * 3
        masks_rim = tf.math.equal(mask_sums, 255 * 3)
        # Cup is max brightness on the blue channel. Revisit this if more
        # classes are added.
        masks_cup = tf.math.equal(mask_sums, 255)
        masks = tf.stack([masks_background, masks_rim, masks_cup], axis=-1)
        masks = tf.cast(masks, tf.uint8)

        return images, masks

    def _resize_data(self, image, mask):
        """
        Resizes images to specified size.

        Using nearest neighbor so dtype remains uint8

        https://www.tensorflow.org/api_docs/python/tf/image/resize

        The return value has type float32, unless the method is
        ResizeMethod.NEAREST_NEIGHBOR, then the return dtype
        is the dtype of images


        We'll convert to the original dtype from tf.io.decode_jpeg
        uint8
        Easier when working with the augmentations
        """
        #image = tf.image.resize(image, (512,512), method='nearest')
        image = tf.image.resize(image, self._image_size, method='nearest')
        mask = tf.image.resize(mask, self._image_size, method='nearest')

        image = tf.cast(image, dtype=tf.uint8)
        mask = tf.cast(mask, dtype=tf.uint8)

        return image, mask

    #############
    def _brightness_random(self, image, mask):
        if self._gen_random_bool():
            brightness_shift = self._random_generator.uniform(
                shape=[1], minval=-50, maxval=50, dtype=tf.dtypes.int32)
            # Do math using signed int32 to avoid overflow/underflow
            image = tf.cast(image, tf.dtypes.int32)
            image += brightness_shift
            image = tf.clip_by_value(image, 0, 255)
            image = tf.cast(image, tf.dtypes.uint8)

        return image, mask

    #############

    def _shift_on_axis(self, image, mask, axis):
        """Shifts images by a random amount on an axis.
        The edges of the image wrap around, they are not filled in with black.
        There also exists a tf.keras.preprocessing.image.random_shift, but
        funnily enough it doesn't like the graph mode used by tf data.

        Arguments:
            axis: 0 or 1. 0 for vertical shift, 1 for horizontal shift.
        """
        max_shift = tf.cast(
            tf.math.floor(self._shift_fraction * self._image_size[1 - axis]),
            tf.dtypes.int32)

        shift_amount = self._random_generator.uniform(shape=[
            1,
        ],
                                                      minval=-max_shift,
                                                      maxval=max_shift,
                                                      dtype=tf.int32)
        # Roll requires a scalar, so the 1D tensor needs to be extracted
        shift_amount = shift_amount[0]
        image = tf.roll(image, shift_amount, axis=axis)
        mask = tf.roll(mask, shift_amount, axis=axis)
        return image, mask

    #############
    def _shift_ver_random(self, image, mask):
        if self._gen_random_bool():
            image, mask = self._shift_on_axis(image, mask, axis=0)
        return image, mask

    #############

    #############
    def _shift_hor_random(self, image, mask):
        if self._gen_random_bool():
            image, mask = self._shift_on_axis(image, mask, axis=1)
        return image, mask

    #############

    #############
    def _rotation_random_cv2(self, image, mask):
        angle = np.random.randint(-20, 20)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((int(w / 2), int(h / 2)), angle, 1)
        image = cv2.warpAffine(image.numpy(), M, (w, h))
        mask = cv2.warpAffine(mask.numpy(), M, (w, h))
        return image, mask

    def _rotate_random(self, image, mask):
        if self._gen_random_bool():
            image, mask = tf.py_function(self._rotation_random_cv2,
                                         [image, mask], [tf.uint8, tf.uint8])

        return image, mask

    #############

    #############
    def _central_crop(self, image, mask):
        if self._gen_random_bool():
            # Use numpy rng as central_crop can't receive a tensor in graph
            # mode.
            central_fraction = 1 - np.random.uniform(0, 1 - self._crop_percent)
            image = tf.image.central_crop(image, central_fraction)
            mask = tf.image.central_crop(mask, central_fraction)
            image, mask = self._resize_data(image, mask)
        return image, mask

    #############

    #############
    def _apply_clahe_cv2(self, image, mask):
        image = image.numpy()
        # Assuming color is in RGB since read in by tf.image.decode_jpeg
        img_hist = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

        # create a CLAHE object (Arguments are optional).
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_hist[:, :, 0] = clahe.apply(img_hist[:, :, 0])

        # convert the YUV image back to RGB format
        image = cv2.cvtColor(img_hist, cv2.COLOR_YUV2RGB)

        return image, mask

    def _apply_clahe(self, image, mask):
        image, mask = tf.py_function(self._apply_clahe_cv2, [image, mask],
                                     [tf.uint8, tf.uint8])
        return image, mask

    def _flip_vertical(self, image, mask):
        seed = self._random_generator.uniform_full_int(shape=[2],
                                                       dtype=tf.dtypes.int32)
        image = tf.image.stateless_random_flip_left_right(image, seed)
        mask = tf.image.stateless_random_flip_left_right(mask, seed)
        return image, mask

    def _flip_horizontal(self, image, mask):
        seed = self._random_generator.uniform_full_int(shape=[2],
                                                       dtype=tf.dtypes.int32)
        image = tf.image.stateless_random_flip_up_down(image, seed)
        mask = tf.image.stateless_random_flip_up_down(mask, seed)
        return image, mask

    def _fixup_shape(self, image, mask):
        """Restore shape info stripped by augmentations."""
        image.set_shape(
            [self._image_size[0], self._image_size[1], self._channels[0]])
        mask.set_shape(
            [self._image_size[0], self._image_size[1], self._channels[1]])
        mask = tf.cast(mask, tf.dtypes.float32)

        return image, mask

    def apply_preprocessing(self, image, mask):
        """Apply backbone-specific preprocessing."""
        image = tf.cast(image, tf.dtypes.float32)
        image = self.preprocessing(image)

        return image, mask

    #############

    def create_data_set(self,
                        CLAHE=False,
                        crop=False,
                        rotate=False,
                        shift_vertical=False,
                        shift_horiztonal=False,
                        flip_horizontal=False,
                        flip_vertical=False,
                        brightness=False):
        '''
        from pathlib import Path
        import glob

         def parse_data( image_paths, mask_paths):
            """
            Reads image and mask files depending on
            specified extension.
            """
            image_content = tf.io.read_file(image_paths)
            mask_content = tf.io.read_file(mask_paths)

            images = tf.image.decode_jpeg(image_content, channels=3)
            masks = tf.image.decode_jpeg(mask_content, channels=1)

            return images, masks

        def resize_data(image, mask):
            """
            Resizes images to specified size.
            """
            image = tf.image.resize(image, (128,128))
            mask = tf.image.resize(mask, (128,128), method='nearest')

            return image, mask

        Sample input:
            image_paths = glob.glob(str(Path('images/split__512_512__8-1-1/train/raw/*.jpg')))
            mask_paths = glob.glob(str(Path('images/split__512_512__8-1-1/train/mask/*.jpg')))
            data = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
            data = data.map(parse_data, num_parallel_calls=tf.data.AUTOTUNE)
            data = data.map(resize_data, num_parallel_calls=tf.data.AUTOTUNE)

        '''

        data = tf.data.Dataset.from_tensor_slices(
            (self._image_paths, self._mask_paths))

        data = data.map(self._parse_data, num_parallel_calls=tf.data.AUTOTUNE)
        # data = data.map(self._resize_data,
        #                 num_parallel_calls=tf.data.AUTOTUNE).cache()
        data = data.cache()
        data = data.shuffle(700)

        if crop:
            data = data.map(self._central_crop,
                            num_parallel_calls=tf.data.AUTOTUNE)

        if shift_vertical:
            data = data.map(self._shift_ver_random,
                            num_parallel_calls=tf.data.AUTOTUNE)

        if shift_horiztonal:
            data = data.map(self._shift_hor_random,
                            num_parallel_calls=tf.data.AUTOTUNE)

        if rotate:
            data = data.map(self._rotate_random,
                            num_parallel_calls=tf.data.AUTOTUNE)

        if flip_horizontal:
            data = data.map(self._flip_horizontal,
                            num_parallel_calls=tf.data.AUTOTUNE)

        if flip_vertical:
            data = data.map(self._flip_vertical,
                            num_parallel_calls=tf.data.AUTOTUNE)

        if brightness:
            data = data.map(self._brightness_random,
                            num_parallel_calls=tf.data.AUTOTUNE)

        if CLAHE:
            data = data.map(self._apply_clahe,
                            num_parallel_calls=tf.data.AUTOTUNE)

        # Apply preprocessing at the end, as many of the augmentation
        # steps expect ints in the range [0-255]
        if self.preprocessing:
            data = data.map(self.apply_preprocessing,
                            num_parallel_calls=tf.data.AUTOTUNE)

        data = data.map(self._fixup_shape)

        return (data)


if __name__ == "__main__":
    import glob
    import os
    import pathlib
    import random

    import cv2
    print(cv2.__version__)
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    import numpy as np
    import tensorflow as tf
    print(tf.__version__)
    physical_devices = tf.config.list_physical_devices('GPU')
    import cv2
    print("Num GPUs:", len(physical_devices))

    image_paths = glob.glob(
        str(pathlib.Path('../images/1.20.21 Split Rim - READY/val/raw/*.jpg')))
    mask_paths = glob.glob(
        str(pathlib.Path(
            '../images/1.20.21 Split Rim - READY/val/mask/*.jpg')))

    np.random.seed(2021)
    tf.random.set_seed(2021)
    random.seed(2021)
    data_loader = DataLoaderDataset(image_paths, mask_paths, (512, 512),
                                    (3, 1))
    data = data_loader.create_data_set(CLAHE=True,
                                       crop=True,
                                       rotate=True,
                                       shift_vertical=True,
                                       shift_horiztonal=True,
                                       brightness=True)

    #############################################
    gs1 = gridspec.GridSpec(1, 2)
    gs1.update(wspace=0.025, hspace=0.05)
    img_index = -1
    i = 0
    for image, mask in data.take(2):
        if i == 0:
            i += 1
            continue

        img_index += 1
        ax1 = plt.subplot(gs1[img_index])
        plt.imshow(image)
        plt.axis('off')
        ax1.set_aspect('equal')

        img_index += 1
        ax = plt.subplot(gs1[img_index])
        plt.imshow(np.squeeze(np.array(mask)), cmap='gray')
        plt.axis('off')
        ax.set_aspect('equal')
    '''
    for img in range(3):
        print(img)
        img_index = -1
        fig = plt.figure(figsize = (10,10))
        gs1 = gridspec.GridSpec(4, 4)
        gs1.update(wspace=0.025, hspace=0.05)
        for i in range(8):
            for image, mask in data.take(1):
                img_index += 1
                ax1 = plt.subplot(gs1[img_index])
                plt.imshow(np.uint8(image))
                plt.axis('off')
                ax1.set_aspect('equal')

                img_index += 1
                ax = plt.subplot(gs1[img_index])
                plt.imshow(np.squeeze(np.array(mask)), cmap='gray')
                plt.axis('off')
                ax1.set_aspect('equal')
        fig.savefig('sample_augmented_dataset_' + str(img+1) + '.jpg', dpi=1200)
        plt.close(fig)
    '''
    #############################################

    if False:

        def apply_clahe(img):
            img_hist = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

            # create a CLAHE object (Arguments are optional).
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_hist[:, :, 0] = clahe.apply(img_hist[:, :, 0])

            # convert the YUV image back to RGB format
            img_hist = cv2.cvtColor(img_hist, cv2.COLOR_YUV2BGR)

            return img_hist

        def plot_image(img_1, img_2, fig, ax, col, plot_title):
            # plt.subplot(nrows, ncols, i)
            ax[0, col].imshow(cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB))
            ax[0, col].axis('off')
            ax[0, col].set_title(plot_title, fontsize=3)

            ax[1, col].imshow(cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB))
            ax[1, col].axis('off')

            ax[2,
               col].imshow(cv2.cvtColor(apply_clahe(img_1), cv2.COLOR_BGR2RGB))
            ax[2, col].axis('off')

            ax[3,
               col].imshow(cv2.cvtColor(apply_clahe(img_2), cv2.COLOR_BGR2RGB))
            ax[3, col].axis('off')

        ###########################
        # View augmentation ranges
        i = 1
        nrows = 4
        ncols = 8
        img_path = 'UCLA_1.3.6.1.4.1.29565.1.4.0.14966.1455033109.175976.jpg'
        img = cv2.imread(img_path)
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)

        # Store height and width of the image
        height, width = img.shape[:2]

        img_mask = cv2.imread(
            'UCLA_1.3.6.1.4.1.29565.1.4.0.14966.1455033109.175976_mask.jpg')
        fig, ax = plt.subplots(nrows, ncols)
        #fig.tight_layout()
        plot_image(img, img, fig, ax, 0, "No Augmentation")

        # Central Crop
        min_value, max_value = (0, 0.6)
        img_aug = tf.image.central_crop(img,
                                        central_fraction=max_value).numpy()
        plot_image(img, img_aug, fig, ax, 1,
                   f"Central Crop {min_value} : {max_value}")

        # Random Rotation
        min_value, max_value = (-30, 30)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((int(w / 2), int(h / 2)), -30, 1)
        img_aug_min = cv2.warpAffine(img, M, (w, h))
        M = cv2.getRotationMatrix2D((int(w / 2), int(h / 2)), 30, 1)
        img_aug_max = cv2.warpAffine(img, M, (w, h))
        plot_image(img_aug_min, img_aug_max, fig, ax, 2,
                   f"Rotation {min_value} : {max_value}")

        # Random Shift Horizontal
        min_value, max_value = (10, 10)
        hor_shift = height / min_value
        T = np.float32([[1, 0, -hor_shift], [0, 1, 1]])
        img_aug_min = cv2.warpAffine(img, T, (width, height))
        T = np.float32([[1, 0, hor_shift], [0, 1, 1]])
        img_aug_max = cv2.warpAffine(img, T, (width, height))
        #plt.imshow(img_aug_min[:,:,::-1])
        #plt.imshow(img_aug_max[:,:,::-1])
        plot_image(img_aug_min, img_aug_max, fig, ax, 3,
                   f"Random Shift H {int(hor_shift)}")

        # Random Shift Vertical
        min_value, max_value = (10, 10)
        ver_shift = height / min_value
        T = np.float32([[1, 0, 1], [0, 1, -ver_shift]])
        img_aug_min = cv2.warpAffine(img, T, (width, height))
        T = np.float32([[1, 0, 1], [0, 1, ver_shift]])
        img_aug_max = cv2.warpAffine(img, T, (width, height))
        #plt.imshow(img_aug_min[:,:,::-1])
        #plt.imshow(img_aug_max[:,:,::-1])
        plot_image(img_aug_min, img_aug_max, fig, ax, 4,
                   f"Random Shift V {int(ver_shift)}")

        # Brightness
        min_value, max_value = (-.2, 0.2)
        img_aug_min = tf.image.adjust_brightness(img, delta=min_value).numpy()
        img_aug_max = tf.image.adjust_brightness(img, delta=max_value).numpy()
        plt.imshow(img_aug_min[:, :, ::-1])
        plot_image(img_aug_min, img_aug_max, fig, ax, 5,
                   f"Brightness {min_value} : {max_value}")

        # Adjust Hue
        min_value, max_value = (-0.05, 0.05)
        img_aug_min = tf.image.adjust_hue(img, delta=min_value,
                                          name=None).numpy()
        img_aug_max = tf.image.adjust_hue(img, delta=max_value,
                                          name=None).numpy()
        plot_image(img_aug_min, img_aug_max, fig, ax, 6,
                   f"Hue {min_value} : {max_value}")

        # Adjust Saturation
        min_value, max_value = (0.5, 1)
        img_aug_min = tf.image.adjust_saturation(img,
                                                 saturation_factor=min_value,
                                                 name=None).numpy()
        img_aug_max = tf.image.adjust_saturation(img,
                                                 saturation_factor=max_value,
                                                 name=None).numpy()
        plot_image(img_aug_min, img_aug_max, fig, ax, 7,
                   f"Saturation {min_value} : {max_value}")

        fig.savefig('sample_augmented.jpg', dpi=1200)
        plt.close(fig)
        ###########################

        ###########################
        # Testing randomness
        '''
        https://machinelearningmastery.com/reproducible-results-neural-networks-keras/

        Keras does get its source of randomness from the NumPy random number generator,
        so this must be seeded regardless of whether you are using a Theano or TensorFlow backend.

        In addition, TensorFlow has its own random number generator that must also be seeded
        '''
        img = cv2.imread(
            'UCLA_1.3.6.1.4.1.29565.1.4.0.14966.1455033109.175976.jpg')
        img_mask = cv2.imread(
            'UCLA_1.3.6.1.4.1.29565.1.4.0.14966.1455033109.175976_mask.jpg')

        fig, ax = plt.subplots(4, 4)

        row = -1
        col = 0
        for i in range(16):
            if i % 4 == 0:
                row += 1
                col = 0
            else:
                col += 1

            if i % 2 == 0:
                img_ = img
            else:
                img_ = img_mask

            #print(row, col, sep = ' : ')
            np.random.seed(row + 5)
            tf.random.set_seed(row + 5)
            img_aug = tf.keras.preprocessing.image.random_rotation(
                img_, rg=60, row_axis=0, col_axis=1, channel_axis=2)

            ax[row, col].imshow(cv2.cvtColor(img_aug, cv2.COLOR_BGR2RGB))
            ax[row, col].axis('off')
        ###########################

        ###########################
        # Test dataset loader
        def parse_data(image_paths, mask_paths):
            """
            Reads image and mask files depending on
            specified extension.

            https://www.tensorflow.org/api_docs/python/tf/io/decode_jpeg

            tf.io.decode_jpeg
            Decode a JPEG-encoded image to a uint8 tensor.
            """
            image_content = tf.io.read_file(image_paths)
            mask_content = tf.io.read_file(mask_paths)

            images = tf.image.decode_jpeg(image_content, channels=3)
            masks = tf.image.decode_jpeg(mask_content, channels=1)

            return images, masks

        def resize_data(image, mask):
            """
            Resizes images to specified size.

            Using nearest neighbor so dtype remains uint8

            https://www.tensorflow.org/api_docs/python/tf/image/resize

            The return value has type float32, unless the method is
            ResizeMethod.NEAREST_NEIGHBOR, then the return dtype
            is the dtype of images

            In our case: uint8 tensor

            Or we can convert to unit8
            """
            #image = tf.image.resize(image, (512,512), method='nearest')
            image = tf.image.resize(image, (512, 512))
            mask = tf.image.resize(mask, (512, 512))

            image = tf.cast(image, dtype=tf.uint8)
            mask = tf.cast(mask, dtype=tf.uint8)
            return image, mask

        #############
        def shift_hor_random_cv2(image, mask):
            height, width = image.shape[:2]
            hor_shift = width / np.random.randint(10, 20)
            T = np.float32([[1, 0, random.choice([-1, 1]) * hor_shift],
                            [0, 1, 1]])

            # We use warpAffine to transform
            # the image using the matrix, T
            image = cv2.warpAffine(image.numpy(), T, (width, height))
            mask = cv2.warpAffine(mask.numpy(), T, (width, height))
            return image, mask

        def shift_hor_random(image, mask):
            if np.random.choice([0, 1]) == 0:
                return image, mask

            image, mask = tf.py_function(shift_hor_random_cv2, [image, mask],
                                         [tf.uint8, tf.uint8])
            return image, mask

        #############

        #############
        def shift_ver_random_cv2(image, mask):
            height, width = image.shape[:2]
            ver_shift = height / np.random.randint(10, 20)
            T = np.float32([[1, 0, 1],
                            [0, 1, random.choice([-1, 1]) * ver_shift]])

            # We use warpAffine to transform
            # the image using the matrix, T
            image = cv2.warpAffine(image.numpy(), T, (width, height))
            mask = cv2.warpAffine(mask.numpy(), T, (width, height))
            return image, mask

        def shift_ver_random(image, mask):
            if np.random.choice([0, 1]) == 0:
                return image, mask

            image, mask = tf.py_function(shift_ver_random_cv2, [image, mask],
                                         [tf.uint8, tf.uint8])
            return image, mask

        #############

        #############
        def rotation_random_cv2(image, mask):
            angle = np.random.randint(-20, 20)
            h, w = image.shape[:2]
            M = cv2.getRotationMatrix2D((int(w / 2), int(h / 2)), angle, 1)
            image = cv2.warpAffine(image.numpy(), M, (w, h))
            mask = cv2.warpAffine(mask.numpy(), M, (w, h))
            return image, mask

        def rotate_random(image, mask):
            if np.random.choice([0, 1]) == 0:
                return image, mask

            image, mask = tf.py_function(rotation_random_cv2, [image, mask],
                                         [tf.uint8, tf.uint8])
            return image, mask

        #############

        #############
        def corrupt_contrast(image, mask):
            """
            Randomly applies a random contrast change.
            """
            print(type(image))
            cond_contrast = tf.cast(
                tf.random.uniform([], maxval=2, dtype=tf.int32), tf.bool)
            image = tf.cond(cond_contrast,
                            lambda: tf.image.random_contrast(image, 0.5, 1.5),
                            lambda: tf.identity(image))
            print(type(image))
            return image, mask

        #############

        #############
        def brightness_random_cv2(image, mask):
            #image = np.clip(image.numpy()+np.random.randint(-50, 50),0,255)
            image = np.uint8(
                np.clip(
                    np.int32(image.numpy()) + np.random.randint(-50, 50), 0,
                    255))
            return image, mask

        def brightness_random(image, mask):
            # if np.random.choice([0, 1]) == 0:
            #  return image, mask
            image, mask = tf.py_function(brightness_random_cv2, [image, mask],
                                         [tf.uint8, tf.uint8])

            #image = tf.image.random_brightness(image, max_delta = 1)
            return image, mask

        def corrupt_brightness(image, mask):
            """
            Radnomly applies a random brightness change.
            """
            cond_brightness = tf.cast(
                tf.random.uniform([], maxval=2, dtype=tf.int32), tf.bool)
            image = tf.cond(cond_brightness,
                            lambda: tf.image.random_brightness(image, 0.5),
                            lambda: tf.identity(image))
            return image, mask

        #############

        #############
        def central_crop_eager(image, mask):
            crop_factor = np.random.randint(65, 100) / 100
            image = tf.image.central_crop(image, central_fraction=crop_factor)
            mask = tf.image.central_crop(mask, central_fraction=crop_factor)
            return image, mask

        def central_crop(image, mask):
            image, mask = tf.py_function(central_crop_eager, [image, mask],
                                         [tf.uint8, tf.uint8])
            #mask = tf.image.central_crop(mask, central_fraction=0.5)
            return image, mask

        #############

        #############
        def apply_clahe_cv2(image, mask):
            image = image.numpy()
            # Assuming color is in RGB since read in by tf.image.decode_jpeg
            img_hist = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

            # create a CLAHE object (Arguments are optional).
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_hist[:, :, 0] = clahe.apply(img_hist[:, :, 0])

            # convert the YUV image back to RGB format
            image = cv2.cvtColor(img_hist, cv2.COLOR_YUV2RGB)

            return image, mask

        def apply_clahe(image, mask):
            image, mask = tf.py_function(apply_clahe_cv2, [image, mask],
                                         [tf.uint8, tf.uint8])
            return image, mask

        #############

        #############################################
        # Make sure the data was loaded correctly by plotting some samples
        '''
        plt.figure(figsize=(4,20))
        img_index = 0
        for i in range(80):
            print(i)
            for image, mask in data.take(1):
                img_index += 1
                plt.subplot(20,8,img_index)
                plt.imshow(np.uint8(image))
                plt.axis('off')

                img_index += 1
                plt.subplot(20,8,img_index)
                plt.imshow(np.squeeze(np.array(mask)), cmap='gray')
                plt.axis('off')

        fig.savefig('sample_augmented_dataset.jpg', dpi=1200)
        plt.close(fig)
        '''
        #############################################

        data = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
        data = data.map(parse_data,
                        num_parallel_calls=tf.data.AUTOTUNE).repeat()
        data = data.map(resize_data, num_parallel_calls=tf.data.AUTOTUNE)
        #data = data.map(brightness_random, num_parallel_calls=tf.data.AUTOTUNE)
        data = data.map(central_crop, num_parallel_calls=tf.data.AUTOTUNE)
        #
        #data = data.map(corrupt_contrast, num_parallel_calls=tf.data.AUTOTUNE)
        #data = data.map(rotate_random, num_parallel_calls=tf.data.AUTOTUNE)
        #data = data.map(shift_ver_random, num_parallel_calls=tf.data.AUTOTUNE)
        #data = data.map(shift_hor_random, num_parallel_calls=tf.data.AUTOTUNE)
        data = data.map(apply_clahe, num_parallel_calls=tf.data.AUTOTUNE)

        #############################################
        img_index = 0
        gs1 = gridspec.GridSpec(1, 2)
        gs1.update(wspace=0.025, hspace=0.0)
        for image, mask in data.take(1):
            ax1 = plt.subplot(gs1[img_index])
            #plt.imshow(np.uint8(image))
            plt.imshow(image)
            plt.axis('off')
            ax1.set_aspect('equal')

            img_index += 1
            ax1 = plt.subplot(gs1[img_index])
            plt.imshow(np.squeeze(np.array(mask)), cmap='gray')
            plt.axis('off')
            ax1.set_aspect('equal')
    #############################################
