import os
import numpy as np

from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.layers import BatchNormalization, Dropout, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from Performance_Matrix import plot_training_history

train_dir = '/home/nihar/PycharmProjects/Emotion Detection/train'
test_dir = '/home/nihar/PycharmProjects/Emotion Detection/test'

# Data augmentation configuration for training data
train_datagen = ImageDataGenerator(
    rescale = 1/255,                # Rescale pixel values to [0,1]
    rotation_range = 40,            # Randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range = 0.2,        # Randomly horizontally shift images
    height_shift_range = 0.2,       # Randomly vertically shift images
    shear_range = 0.2,              # Apply shearing transformations
    zoom_range = 0.1,               # Randomly zoom image
    horizontal_flip = True,         # Randomly flip images horizontally
    fill_mode = 'nearest'           # Strategy used for filling in newly created pixels
)

# Rescaling for validation / test data (without further data augmentation)
test_datagen = ImageDataGenerator(
    rescale = 1/255           # Rescale pixel values to [0, 1]
)

# Creating data genetarors for training
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size = (224, 224),     # Resize images to 224x224 for model input
    color_mode = 'rgb',           # Images will be converted to RGB
    class_mode = 'categorical',   # For multi-class classification
    batch_size = 32               # Size of the batches of data
)

# Creating data generators for testing/validation
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size = (224, 224),        # Resize images to 224x224 for model input
    color_mode = 'rgb',              # Images will be converted to RGB
    class_mode = 'categorical',      # For multi-class classification
    batch_size = 32                  # Size of the batches of data
)


# Extract class labels for all instances in the training dataset
classes = np.array(train_generator.classes)

# Calculated class weights to handle imbalances in the training data
# 'balanced' mode automatically adjusts weights inversely proportional to class frequencies
class_weights = compute_class_weight(
    class_weight = 'balanced',        # Strategy to balance classes
    classes = np.unique(classes),     # Unique class labels
    y = classes                       # Class labels for each instance in the training dataset
)

# Create a dictionary mapping class indices to their calculated weights
class_weights_dict = dict(enumerate(class_weights))

# Output the class weights dictionary
print("Class Weights Dictionary:", class_weights_dict)



input_shape = (224, 224, 3)

base_model = ResNet50V2(include_top = False, weights = 'imagenet', input_shape = input_shape)

model = Sequential([
    base_model,
    BatchNormalization(),
    GlobalAveragePooling2D(),
    Dense(512, activation = 'relu'),
    Dropout(0.1),
    Dense(512, activation = 'relu'),
    Dropout(0.1),
    Dense(128, activation = 'relu'),
    Dropout(0.1),
    Dense(7, activation = 'softmax')
])

optimizer = Adamax(learning_rate = 0.0001)

model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.summary()


# File path for the model checkpoint
cnn_path = '/home/nihar/PycharmProjects/Emotion Detection/ResNet50_Transfer_Learning'
name = 'ResNet50_Transfer_Learning.keras'
chk_path = os.path.join(cnn_path, name)

# Callback to save the model checkpoint
checkpoint = ModelCheckpoint(filepath = chk_path,
                             save_best_only = True,
                             verbose = 1,
                             mode = 'min',
                             monitor = 'val_loss')

# Callback for early stopping
earlystop = EarlyStopping(monitor = 'val_loss',
                          min_delta = 0,
                          patience = 10,
                          verbose = 1,
                          restore_best_weights = True)

# Callback to reduce learning rate
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
                              factor = 0.2,
                              patience = 6,
                              verbose = 1,
                              min_delta = 0.0001)

# Callback to log training data to a csv file
csv_logger = CSVLogger(os.path.join(cnn_path, 'training.log'))

# Aggregating all callbacks into a list
callbacks = [checkpoint, earlystop, csv_logger]   # Adjust as per your use-case

train_steps_per_epoch = train_generator.samples // train_generator.batch_size + 1
# validation_steps_epoch = validation_generator.samples // validation_generator.batch_size + 1
test_steps_epoch = test_generator.samples // test_generator.batch_size + 1

# Load your pre-trained model
# model = tf.keras.models.load_model('/home/nihar/PycharmProjects/Emotion Detection/ResNet50_Transfer_Learning.keras')

history = model.fit(
    train_generator,
    steps_per_epoch = 200,
    epochs = 50,
    validation_data = test_generator,
    validation_steps = 100,
    class_weight = class_weights_dict,
    callbacks = callbacks
)

print(history.history)

plot_training_history(history)