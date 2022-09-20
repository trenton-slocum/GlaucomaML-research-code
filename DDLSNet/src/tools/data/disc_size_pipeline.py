import numpy as np
import pandas as pd
import tensorflow as tf


class DiscSizeDSBuilder:
    """This class builds TF Datasets for the disc size prediction task.

    Args:
        image_directory: Path to where all images reside
        image_size: List of [W, H] representing the input size of the model
        num_classes: Number of possible classes (e.g. 3 for S/M/L)
        label_name: String name of the label column.
    """

    def __init__(self, image_directory, image_size, num_classes, label_name):
        self.image_directory = image_directory
        self.image_size = image_size
        self.num_classes = num_classes
        self.label_name = label_name

    def parse_image(self, filename, label):
        """Read in, decode, and resize an input image."""
        image_path = self.image_directory + '/' + filename
        image = tf.io.read_file(image_path)
        # Downscale in the decode for speed
        # This should take our smallest images, the cropped ones,
        # down to 250x250
        # Image will be ints in the range 0-255
        image = tf.image.decode_jpeg(image, ratio=4)
        # Image will be floats in the range 0-255
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, self.image_size)
        return image, label

    def encode_label(self, filename, label):
        """Converts a label of 1 to [1, 0, 0]."""
        new_label = tf.one_hot(label - 1, depth=self.num_classes)
        return filename, new_label

    def gen_ds(self, df, batch_size, shuffle):
        """Generates a tf.Data pipeline for the images listed in df.
        Args:
            df: A Pandas.DataFrame containing the columns 'filename' and
                self.label_name. These are used to read in files, process the
                images, and associate them with their labels.
            batch_size: An integer indicating batch size.
            shuffle: A boolean indicating whether items should be shuffled.
                Set to False when evaluating, and True when training.
        Returns: a tf.Data.Dataset containing the processed + shuffled images in
            df.
        """
        files = df['filename'].to_numpy()
        labels = df[self.label_name].to_numpy().astype(np.int32)
        ds = tf.data.Dataset.from_tensor_slices((files, labels))

        if shuffle:
            # This shuffle will only be done once, on the first epoch
            ds = ds.shuffle(len(df))

        # Allow to run in parallel and in non-deterministic order for speed.
        # This speeds up the first epoch significantly.
        ds = ds.map(self.parse_image,
                    num_parallel_calls=tf.data.AUTOTUNE,
                    deterministic=False)
        ds = ds.map(self.encode_label)
        # Cache to avoid repeating the slow read + decode + resize
        ds = ds.cache()

        if shuffle:
            # Add another shuffle to keep things dynamic each epoch
            ds = ds.shuffle(1000)

        ds = ds.batch(batch_size)

        return ds
