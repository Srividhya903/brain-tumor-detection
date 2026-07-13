# Import required libraries
# Import required libraries
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing import image

# Define the CNN model
model = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(units=64, activation='relu'),
    Dropout(0.5),
    Dense(units=1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Prepare the data
train_data_gen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_data_gen = ImageDataGenerator(rescale=1./255)
training_set = train_data_gen.flow_from_directory('C:/Users/acer/OneDrive/Desktop/Projects@TWM/Brain-Tumor-Detection-master/xray/train', 
                                                   target_size=(128, 128), 
                                                   batch_size=32, 
                                                   class_mode='binary',
                                                   classes=['Normal', 'Tumor'])
test_set = test_data_gen.flow_from_directory('C:/Users/acer/OneDrive/Desktop/Projects@TWM/Brain-Tumor-Detection-master/xray/test', 
                                              target_size=(128, 128), 
                                              batch_size=32, 
                                              class_mode='binary',
                                              classes=['Normal', 'Tumor'])
validation_set = test_data_gen.flow_from_directory('C:/Users/acer/OneDrive/Desktop/Projects@TWM/Brain-Tumor-Detection-master/xray/validation', 
                                              target_size=(128, 128), 
                                              batch_size=32, 
                                              class_mode='binary',
                                              classes=['Normal', 'Tumor'])

# Train the model and plot accuracy and loss graphs
history = model.fit(training_set, epochs=10, validation_data=test_set)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_set)
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)

# Load the saved model
model = tf.keras.models.load_model('brain_tumor_detection.h5')

# Make a prediction on a single image
img_path = 'C:/Users/acer/OneDrive/Desktop/Projects@TWM/Brain-Tumor-Detection-master/Testing-Images/Tumor/Y14.jpg'
img = image.load_img(img_path, target_size=(128, 128))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
images = np.vstack([x])
classes = model.predict(images, batch_size=10)
print("It is '{}' ".format('Tumor' if classes[0]>0.5 else 'Normal'))
