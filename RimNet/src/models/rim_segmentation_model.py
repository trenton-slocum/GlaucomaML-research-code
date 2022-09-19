# coding: utf-8
"""
Script to train segmentation model described in paper.
"""
import segmentation_models as sm

sm.set_framework('tf.keras')

from functools import partial

import keras_tuner as kt
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import ReduceLROnPlateau

print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

import datetime
import glob
import pathlib
import time

import data_loader_dataset
import image_handler
import numpy as np

# mixed_precision.set_global_policy('mixed_float16')

IMAGE_SIZE = [224, 224]
NUM_INPUT_CHANNELS = [3]
# One of 'DM' for disks, 'Rim' for whole rims and 'rim_broken' for broken
SEGMENTATION_TYPE = 'Rim-multiclass'

IMAGE_PATH_PREFIX = '../images/4.24.22 Broken-Whole Rim Ready'

# split_CM_Trial 8 and DM_Trial 8 are the old one
# Whether to run with the CUPTI profiler
PROFILE = True

# Name of tuner, either 'randomsearch' or 'hyperband'
TUNER = 'randomsearch'

# backbones to consider
BACKBONES = [
    # 'mobilenetv2',
    # 'resnet34',
    # 'efficientnetb0',
    'inceptionv3',
    # 'resnet101',
    # 'vgg16',
    # 'efficientnetb3',
    # 'resnet50',
]

ARCHITECTURES = [
    'unet',
    # 'fpn',
    'linknet',
    # 'pspnet',
]

LOSSES = ['categorical_crossentropy', 'categorical_focal_loss']
OPTIMIZERS = ['adam']

# Whether to run the copy phase of image handler
COPY_IMAGES = False

####################################################
START = 0


def time_start():
    global START
    START = time.perf_counter()


def time_end():
    global START
    end = time.perf_counter() - START

    print(time.strftime('%H:%M:%S', time.gmtime(end)))


# Create train dataset
# These must be sorted to ensure filenames line up. Extensions are at the end so they
# Shouldn't matter
image_paths = sorted(
    glob.glob(str(pathlib.Path(f"{IMAGE_PATH_PREFIX}/train/raw/*.png"))))
mask_paths = sorted(
    glob.glob(str(pathlib.Path(f"{IMAGE_PATH_PREFIX}/train/mask/*.png"))))

data_loader = data_loader_dataset.DataLoaderDataset(image_paths,
                                                    mask_paths,
                                                    IMAGE_SIZE, (3, 3),
                                                    preprocessing=None)
dataset = data_loader.create_data_set(CLAHE=True,
                                      crop=True,
                                      rotate=True,
                                      shift_vertical=True,
                                      shift_horiztonal=True,
                                      flip_horizontal=True,
                                      flip_vertical=True,
                                      brightness=True)

# Create validation dataset
val_image_paths = sorted(
    glob.glob(str(pathlib.Path(f"{IMAGE_PATH_PREFIX}/val/raw/*.png"))))
val_mask_paths = sorted(
    glob.glob(str(pathlib.Path(f"{IMAGE_PATH_PREFIX}/val/mask/*.png"))))

val_data_loader = data_loader_dataset.DataLoaderDataset(
    val_image_paths, val_mask_paths, IMAGE_SIZE, (3, 3))
val_dataset = val_data_loader.create_data_set(CLAHE=True)

# 1 if only one class (binary), else num_classes + 1 to account for background
# Here we have 2 classes (rim + cup), so 3 output channels
OUTPUT_CHANNELS = 3


def model_builder(hp):
    """This function returns a model made with the given hyperparameters."""
    # Name of backbone to use
    hp_backbone = hp.Choice('backbone', values=BACKBONES)
    hp_loss = hp.Choice('loss_function', values=LOSSES)
    # hp_freeze_encoder = hp.Boolean('freeze_encoder')
    hp_architecture = hp.Choice('architecture', values=ARCHITECTURES)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4])
    hp_optimizer = hp.Choice('optimizer', values=OPTIMIZERS)

    inputs = tf.keras.layers.Input(shape=IMAGE_SIZE + NUM_INPUT_CHANNELS)

    preprocess_input = sm.get_preprocessing(hp_backbone)

    scaled_inputs = preprocess_input(inputs)

    architecture_classes = {
        'unet': sm.Unet,
        'linknet': sm.Linknet,
        'pspnet': partial(sm.PSPNet, downsample_factor=8),
        'fpn': sm.FPN,
    }

    architecture = architecture_classes[hp_architecture]

    segmenter = architecture(
        backbone_name=hp_backbone,
        classes=OUTPUT_CHANNELS,
        input_shape=IMAGE_SIZE + NUM_INPUT_CHANNELS,
        #  encoder_freeze=hp_freeze_encoder,
        encoder_weights='imagenet')

    outputs = segmenter(scaled_inputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    loss_functions = {
        'categorical_crossentropy': sm.losses.categorical_crossentropy,
        'categorical_focal_loss': sm.losses.categorical_focal_loss,
    }

    optimizer_functions = {
        'adam': tf.keras.optimizers.Adam,
        'sgd': tf.keras.optimizers.SGD
    }

    optimizer = optimizer_functions[hp_optimizer]

    model.compile(
        optimizer=optimizer(learning_rate=hp_learning_rate),
        loss=loss_functions[hp_loss],
        metrics=['accuracy', sm.metrics.iou_score],
    )
    return model


batch_size = 32
train_dataset = dataset.batch(batch_size).prefetch(
    tf.data.AUTOTUNE)  # .repeat()
val_batch_dataset = val_dataset.batch(batch_size).prefetch(
    tf.data.AUTOTUNE)  # .repeat()

if TUNER == 'hyperband':
    tuner = kt.Hyperband(
        model_builder,
        objective=kt.Objective("val_iou_score", direction="max"),
        max_epochs=500,
        factor=3,
        hyperband_iterations=2,  # Increase for better results
        directory='hyperband_tuning',
        project_name='segmentation',
        overwrite=True)
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=50,
                                                  restore_best_weights=True)
    tuner.search(train_dataset,
                 validation_data=val_batch_dataset,
                 epochs=500,
                 callbacks=[stop_early])
elif TUNER == 'randomsearch':
    tuner = kt.RandomSearch(model_builder,
                            objective=kt.Objective("val_iou_score",
                                                   direction="max"),
                            overwrite=True,
                            max_trials=80)

    tuner.search(train_dataset, validation_data=val_batch_dataset, epochs=300)

print('done!')

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print('The bests hyperparams are:')
print(best_hps.values)

# Set up and train the best model

model_name = datetime.datetime.now().strftime(
    f'%Y%m%d-%H%M%S-{SEGMENTATION_TYPE}')

# Tensorboard callback to examine loss curves
log_dir = "../new_logs/fit/" + model_name
if PROFILE:
    profile_batch = '50,60'
else:
    profile_batch = 0
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, profile_batch=profile_batch, histogram_freq=10)

# Annealing Learning Rate Approach
# https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ReduceLROnPlateau
# if MeanIoU not improved in 50 epochs, reduce LR by half
learning_rate_reduction = ReduceLROnPlateau(monitor='val_iou_score',
                                            mode='max',
                                            patience=50,
                                            verbose=1,
                                            factor=0.9,
                                            min_lr=0.0001)

###Add checkpoints to save model###
checkpoint_path = "../models/checkpoints" + model_name
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 monitor='val_iou_score',
                                                 mode='max',
                                                 save_best_only=True,
                                                 save_freq='epoch',
                                                 verbose=1)
model = tuner.hypermodel.build(best_hps)

results = model.fit(
    train_dataset,
    validation_data=val_batch_dataset,
    epochs=1000,
    callbacks=[tensorboard_callback, cp_callback, learning_rate_reduction])

model.save(f'../models/{model_name}_forced')

model.load_weights(checkpoint_path)

model.save(f'../models/{model_name}_best')
