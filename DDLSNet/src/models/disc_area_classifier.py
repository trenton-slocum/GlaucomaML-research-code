"""Script to train a model that predicts disc size for image of optic nerve head."""

import numpy as np
import pandas as pd
import tensorflow as tf
from tools.data.disc_size_pipeline import DiscSizeDSBuilder
from tools.metrics import (gen_confusion_matrix, get_accuracy_scores,
                           get_auc_scores, get_classification_report)

# Toggle to either train and save best model, or evaluate best model on test
MODE = 'evaluate'

# Options for labels are GROUPING_3_ALL, GROUPING_15_70_15, and GROUPING_10_ALL
LABEL_NAME = 'GROUPING_15_70_15'
IMAGE_SIZE = [224, 224]
# Boolean indicating whether to use augmentations in training
AUGMENTATION = True
# Metrics shown during training
METRICS = ['accuracy', 'AUC']

LOGFILE = './disc_tuning_fun.txt'
f = open(LOGFILE, 'w')

# Whether to print out the layers of the backbone model
PRINT_LAYERS = False

# List of strings with camera names.
# Can be Slide, Digital01, Digital02, Digital03
CAMERAS = ['Slide', 'Digital01', 'Digital02', 'Digital03']

# One of cropped or uncropped
IMAGE_TYPE = 'cropped'

if IMAGE_TYPE == 'cropped':
    IMAGE_DIRECTORY = '../images/images_cropped_clean_v1'
else:
    IMAGE_DIRECTORY = '../images/Rim Area - PIL - READY'

master_list_file = pd.ExcelFile("./data/disc_area_master_list_stripped.xlsx")

# Options for sheets are Data_All and Data_Single
master_list = pd.read_excel(master_list_file, 'Data_All')

print('All columns available:')
print(master_list.columns)

original_num_items = len(master_list)
# Grab only images from the desired columns
master_list = master_list[master_list['Camera'].isin(CAMERAS)]

print(
    f'Filtered out {original_num_items - len(master_list)} images due to camera'
)

# Grab only cropped images, if desired
if IMAGE_TYPE == 'cropped':
    original_num_items = len(master_list)
    master_list = master_list.query('CROPPED_READY == True')
    print(f'Removed {original_num_items - len(master_list)} without crop')
print(f'Final number of images: {len(master_list)}')

# Grab only the relevant columns
relevant_master_list = master_list[['filename', LABEL_NAME, 'type']]

print('Top of the master list:')
print(relevant_master_list.head(5))

splits = {name: data for (name, data) in relevant_master_list.groupby('type')}

train = splits['Train'].drop(columns=['type'])
test = splits['Test'].drop(columns=['type'])
val = splits['Val'].drop(columns=['type'])


def print_splits(df):
    """Prints out the class splits in the given dataframe."""
    counts = df[LABEL_NAME].value_counts()
    fractions = counts / len(df)

    frame = {'counts': counts, 'fractions': fractions}

    info = pd.DataFrame(frame)
    print(info)


print('Train split')
print_splits(train)
print('Val Split')
print_splits(val)
print('Test Split')
print_splits(test)

print('Top of train:')
print(train.head(5))

# This is the number of unique labels
NUM_CLASSES = relevant_master_list[LABEL_NAME].max()

mirrored_strategy = tf.distribute.MirroredStrategy(
    cross_device_ops=tf.distribute.ReductionToOneDevice())


def build_model(hp):
    backbone = hp['backbone']
    learning_rate = hp['phase_one_lr']

    # Creation of model, optimizer, and metrics must be inside the scope
    with mirrored_strategy.scope():
        # TODO(Tyler): Set up multi GPU https://www.tensorflow.org/tutorials/distribute/keras
        original_inputs = tf.keras.layers.Input(shape=IMAGE_SIZE + [3])

        if backbone == 'inceptionv3':
            base_model = tf.keras.applications.InceptionV3(
                weights='imagenet',
                input_shape=IMAGE_SIZE + [3],
                include_top=False)
            scaler = tf.keras.applications.inception_v3.preprocess_input
        elif backbone == 'efficientnetb4':
            base_model = tf.keras.applications.EfficientNetB4(
                weights='imagenet',
                input_shape=IMAGE_SIZE + [3],
                include_top=False)
            scaler = tf.keras.applications.efficientnet.preprocess_input
        elif backbone == 'efficientnetb0':
            base_model = tf.keras.applications.EfficientNetB0(
                weights='imagenet',
                input_shape=IMAGE_SIZE + [3],
                include_top=False)
            scaler = tf.keras.applications.efficientnet.preprocess_input
        elif backbone == 'resnet101v2':
            base_model = tf.keras.applications.ResNet101V2(
                weights='imagenet',
                input_shape=IMAGE_SIZE + [3],
                include_top=False)
            scaler = tf.keras.applications.resnet_v2.preprocess_input
        elif backbone == 'resnet50v2':
            base_model = tf.keras.applications.ResNet50V2(
                weights='imagenet',
                input_shape=IMAGE_SIZE + [3],
                include_top=False)
            scaler = tf.keras.applications.resnet_v2.preprocess_input
        elif backbone == 'vgg16':
            base_model = tf.keras.applications.VGG16(weights='imagenet',
                                                     input_shape=IMAGE_SIZE +
                                                     [3],
                                                     include_top=False)
            scaler = tf.keras.applications.vgg16.preprocess_input
        elif backbone == 'vgg19':
            base_model = tf.keras.applications.VGG19(weights='imagenet',
                                                     input_shape=IMAGE_SIZE +
                                                     [3],
                                                     include_top=False)
            scaler = tf.keras.applications.vgg19.preprocess_input
        else:
            raise ValueError("Invalid backbone name provided.")

        # TODO(Tyler): Determine best values for these augmentations
        # Augmentations applied only at train time. Put them before the scaler
        if AUGMENTATION:
            inputs = tf.keras.layers.RandomFlip()(original_inputs)
            # Rotate .1 radians left or right, or 18 degrees
            inputs = tf.keras.layers.RandomRotation(factor=.05)(inputs)
            inputs = tf.keras.layers.RandomTranslation(height_factor=.2,
                                                       width_factor=.2)(inputs)
            inputs = tf.keras.layers.RandomZoom(height_factor=.2)(inputs)
            # inputs = tf.keras.layers.RandomContrast(factor=.1)(inputs)
            # Clip to make sure that all are in valid pixel ranges
            inputs = tf.clip_by_value(inputs, 0, 255)
        else:
            inputs = original_inputs

        # Take in 0-255 floats and rescale + preprocess
        scaled_inputs = scaler(inputs)

        x = base_model(scaled_inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        # x = tf.keras.layers.Flatten(name="flatten")(x)
        # let's add a fully-connected layer
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        # x = tf.keras.layers.Dense(1024, activation='relu')(x) ## try a smaller number of units - Zhe
        # x = tf.keras.layers.Dropout(0.5)(x)
        # x = tf.keras.layers.Dense(256, activation='relu')(x) ## try a smaller number of units - Zhe
        # x = tf.keras.layers.Dropout(0.5)(x)
        # x = tf.keras.layers.Dense(64, activation='relu')(x) ## try a smaller number of units - Zhe
        # x = tf.keras.layers.Dropout(0.5)(x)
        # and a logistic layer -- let's say we have DEPTH classes
        predictions = tf.keras.layers.Dense(NUM_CLASSES,
                                            activation='softmax')(x)

        model = tf.keras.Model(inputs=original_inputs, outputs=predictions)

        base_model.trainable = False

        # TODO(Tyler): Determine if rmsprop/SGD should be swapped for ADAM or something
        # compile the model (should be done *after* setting layers to non-trainable)
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                      loss='categorical_crossentropy',
                      metrics=METRICS)

    return model, base_model


def unlock_and_compile_model(model, base_model, hps):
    """Unlock and compile the model according to hps"""
    learning_rate = hps['phase_two_lr']
    optimizer = hps['phase_two_optimizer']
    # Unlock the full model before freezing the layers we don't want to tune
    base_model.trainable = True

    num_layers = len(base_model.layers)
    # Index of layer from which to begin fine tuning
    # e.g. 311 layers, tune_fraction is .2
    # 311 - int(.2 * 311) = 249, we tune the last 20% of layers
    fine_tune_at = num_layers - int(hps['tune_fraction'] * num_layers)
    print(f'Unlocking from layer: {fine_tune_at}')

    with mirrored_strategy.scope():
        # we chose to train the top 2 inception blocks, i.e. we will freeze
        # the first 249 layers and unfreeze the rest:
        base_model.trainable = True
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

        optimizer_functions = {
            'adam': tf.keras.optimizers.Adam,
            'sgd': tf.keras.optimizers.SGD,
            'rmsprop': tf.keras.optimizers.RMSprop,
        }

        # we need to recompile the model for the above modifications to take effect
        model.compile(
            optimizer=optimizer_functions[optimizer](lr=learning_rate),
            loss='categorical_crossentropy',
            metrics=METRICS)

    return model


# TODO(Tyler): Determine optimal batch size. With size 32, only ~32% GPU util
BATCH_SIZE_PER_REPLICA = 32
# Batch gets split across n_gpus, so scale accordingly
batch_size = (BATCH_SIZE_PER_REPLICA * mirrored_strategy.num_replicas_in_sync)
builder = DiscSizeDSBuilder(
    IMAGE_DIRECTORY,
    IMAGE_SIZE,
    NUM_CLASSES,
    LABEL_NAME,
)

train_ds = builder.gen_ds(
    train,
    batch_size=batch_size,
    shuffle=True,
)
val_ds = builder.gen_ds(
    val,
    batch_size=batch_size,
    shuffle=False,
)
test_ds = builder.gen_ds(
    test,
    batch_size=batch_size,
    shuffle=False,
)

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

# TODO(Tyler): Add some logging to tensorboard
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                  patience=70,
                                                  restore_best_weights=True)

# Calculated as (1 / num_class) * (total / NUM_CLASSES)

total_samples = len(train)

# Class names are 1, 2, 3, but the model knows them as 0, 1, 2, thus label - 1
class_weights = {
    label - 1:
    (1 / (train[LABEL_NAME] == label).sum()) * (total_samples / NUM_CLASSES)
    for label in [1, 2, 3]
}

print('Class weights:')
print(class_weights)


def train_model(hps):
    """Builds and trains a model, given hyperparams dictionary"""

    backbone_name = hps['backbone']
    model, base_model = build_model(hps)

    print(f'Total number of layers in {backbone_name}:')
    print(len(base_model.layers))

    model.fit(train_ds,
              validation_data=val_ds,
              epochs=100,
              class_weight=class_weights,
              callbacks=[early_stopping])

    # Validation confusion matrix after partial training
    print('Beginning Initial confusion matrix')
    initial_cm = gen_confusion_matrix(model, val_ds, NUM_CLASSES)
    print(initial_cm)

    # let's visualize layer names and layer indices to see how many layers
    # we should freeze:
    if PRINT_LAYERS:
        for i, layer in enumerate(base_model.layers):
            print(i, layer.name)

    #Add checkpoints to save model
    model_name = f'{backbone_name}_morepics_' + f'augs-{AUGMENTATION}_{IMAGE_SIZE[0]}_'
    checkpoint_path = "../models/checkpoints/" + model_name
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     monitor='val_accuracy',
                                                     save_best_only=True,
                                                     save_freq='epoch',
                                                     verbose=1)

    model = unlock_and_compile_model(model, base_model, hps)
    # we train our model again (this time fine-tuning the top blocks
    # alongside the top Dense layer)
    model.fit(train_ds,
              validation_data=val_ds,
              epochs=500,
              class_weight=class_weights,
              callbacks=[cp_callback])

    # Load back the best performing model
    model.load_weights(checkpoint_path)

    # Save it out in full
    model.save(f'../models/{model_name}_best')

    print('Generating final confusion matrix...')
    print(gen_confusion_matrix(model, val_ds, NUM_CLASSES))

    print('Accuracy scores:')
    accuracy_scores = get_accuracy_scores(model, val_ds)
    for average_type, score in accuracy_scores.items():
        print(f'{average_type} Accuracy score: {score}')

    print('ROC_AUC Scores:')

    auc_scores = get_auc_scores(model, val_ds)

    for average_type, score in auc_scores.items():
        print(f'{average_type} ROC_AUC score: {score}')

    print('Classification report:')

    print(get_classification_report(model, val_ds))

    return model, accuracy_scores['raw']


def random_search_models(search_space, iterations):
    """Randomly sample models from the search space and train them."""
    explored_combos = []
    max_combos = 1

    best_performance = 0
    best_model = None
    best_hps = None

    for values in search_space.values():
        max_combos = max_combos * len(values)

    print(f'Size of search space is {max_combos}')
    print(f'Running for {min(iterations, max_combos)} iterations')

    for i in range(min(iterations, max_combos)):
        print(f'Beginning trial {i}')
        while True:
            new_dict = {}
            for key, value in search_space.items():
                new_dict[key] = np.random.choice(value)

            if new_dict not in explored_combos:
                explored_combos.append(new_dict)
                break
        hps = new_dict

        model, raw_accuracy = train_model(hps)

        if raw_accuracy > best_performance:
            best_performance = raw_accuracy
            best_model = model
            best_model.save('../models/disc_size_tuned_wip-more-pics')
            best_hps = hps
            print('New best performer found:')
            print(best_hps)
            print(f'New performance: {best_performance}')

            f.write('New best performer found:\n')
            f.write(repr(best_hps) + '\n')
            f.write(f'New performance: {best_performance}\n')

    return best_model, best_hps


BACKBONE_NAMES = [
    'inceptionv3', 'efficientnetb4', 'efficientnetb0', 'resnet101v2',
    'resnet50v2', 'vgg16', 'vgg19'
]

PHASE_ONE_LRS = [1e-4, 1e-5]
PHASE_TWO_LRS = [1e-5, 1e-6, 1e-7]
TUNE_FRACTIONS = [.1, .2, .5]
PHASE_TWO_OPTIMIZERS = ['adam', 'sgd', 'rmsprop']

search_space = {
    'backbone': BACKBONE_NAMES,
    'phase_one_lr': PHASE_ONE_LRS,
    'phase_two_lr': PHASE_TWO_LRS,
    'tune_fraction': TUNE_FRACTIONS,
    'phase_two_optimizer': PHASE_TWO_OPTIMIZERS,
}


def gen_full_report(model, evaluation_data=val_ds):
    print('Generating final confusion matrix...')
    print(gen_confusion_matrix(model, evaluation_data, NUM_CLASSES))

    print('Accuracy scores:')
    accuracy_scores = get_accuracy_scores(model, evaluation_data)
    for average_type, score in accuracy_scores.items():
        print(f'{average_type} Accuracy score: {score}')

    print('ROC_AUC Scores:')

    auc_scores = get_auc_scores(model, evaluation_data)

    for average_type, score in auc_scores.items():
        print(f'{average_type} ROC_AUC score: {score}')

    print('Classification report:')

    print(get_classification_report(model, evaluation_data))


if MODE == 'train':
    best_model, best_hps = random_search_models(search_space, 30)

    print('Best hyperparameters')
    print(best_hps)

    gen_full_report(best_model)
    best_model.save('../models/disc_size_tuned_best-more-pics')

elif MODE == 'evaluate':
    best_model = tf.keras.models.load_model(
        '../models/disc_size_tuned_best-more-pics')
    gen_full_report(best_model, test_ds)

f.close()
