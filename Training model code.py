import pandas as pd
import tensorflow as tf
import os
import random
import json
from datetime import datetime
from tensorflow.keras import layers, models, callbacks
import io 
from contextlib import redirect_stdout
import matplotlib.pyplot as plt  # Import Matplotlib for plotting



#The images used for the training model are available at the below-mentioned URL
#https://drive.google.com/drive/folders/1Poa6OMQh7fv3WEUIIOZ_jOmXXcRVTbTx?usp=sharing

# Path to the Excel file containing colony counts
excel_path = r"path of excel file with colony count number"

# Load colony counts from ODS file
def load_colony_counts_from_excel(excel_path):
    df = pd.read_excel(excel_path, engine='openpyxl')
    return df['Image_Name'].tolist(), df['Colony_Count'].tolist() 

# Load the images and their corresponding colony counts
image_paths, colony_counts = load_colony_counts_from_excel(excel_path)

# Folder path where the images are stored
folder_path = r"path to folder containing images"

def load_and_preprocess_image(image_path, augment=False, target_size=(256, 256), extension=".jpg"):
    # Construct the full path including the file extension
    full_path = os.path.join(folder_path, image_path + extension)  # Add the extension here
    print(f"Processing: {full_path}")
    
    # Load image and resize
    img = tf.io.read_file(full_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, target_size)
    
    # Augmentation
    if augment:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, max_delta=0.2)
        img = tf.image.random_contrast(img, lower=0.5, upper=1.5)
    
    # Normalize pixel values to range [0, 1]
    img = tf.cast(img, tf.float32) / 255.0
    return img

# Function to generate augmented images from an original image
def generate_augmented_images(image_path, colony_count):
    augmented_images = []
    for _ in range(5):  # Generate fewer augmented images to reduce data size
        img = load_and_preprocess_image(image_path, augment=True, target_size=(256, 256))
        augmented_images.append((img, colony_count))
    return augmented_images

# Data structure to store augmented images and their labels
augmented_data = []

# Generate augmented images for each original image
for image_path, colony_count in zip(image_paths, colony_counts):
    augmented_data.extend(generate_augmented_images(image_path, colony_count))

# Shuffle the augmented data
random.shuffle(augmented_data)

# Separate images and labels
images = [pair[0] for pair in augmented_data]
labels = [pair[1] for pair in augmented_data]

# Convert images and labels to TensorFlow dataset
dataset = tf.data.Dataset.from_tensor_slices((images, labels))

# Define the CNN model
def create_model(input_shape):
    kernel_size = 3
    pool_size = 2
    model = models.Sequential([
        layers.Conv2D(4, (kernel_size, kernel_size), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((pool_size, pool_size)),
        layers.Conv2D(8, (kernel_size, kernel_size), activation='relu'),
        layers.MaxPooling2D((pool_size, pool_size)),
        layers.Conv2D(16, (kernel_size, kernel_size), activation='relu'),
        layers.MaxPooling2D((pool_size, pool_size)),
        layers.Conv2D(32, (kernel_size, kernel_size), activation='relu'),
        layers.MaxPooling2D((pool_size, pool_size)),
        layers.Conv2D(64, (kernel_size, kernel_size), activation='relu'),
        layers.MaxPooling2D((pool_size, pool_size)),
        layers.Conv2D(128, (kernel_size, kernel_size), activation='relu'),
        layers.MaxPooling2D((pool_size, pool_size)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.2),    
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),   
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),   
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2), 
        layers.Dense(1)  # Output layer with single neuron for regression
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

# Define input image dimensions
input_shape = (256, 256, 3)

# Create the CNN model
model = create_model(input_shape)

# Print model summary
model.summary()

# Split the dataset into training and validation sets
train_size = int(0.8 * len(images))
train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)

# Configure dataset for performance
batch_size=8
train_dataset = train_dataset.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
val_dataset = val_dataset.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# Define early stopping callback
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Train the model
epochs = 5
history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)

# Evaluate the model
loss = model.evaluate(val_dataset)
print("Validation Loss:", loss)

# Save the trained model
model_file_name = os.path.join(
    r"desired path to save the trained model",
    f"epoch_{epochs}_{int(datetime.now().timestamp())}.h5"
)

model.save(model_file_name)


from contextlib import redirect_stdout

# Function to get a string summary of the model
def get_model_summary(model):
    stream = io.StringIO()
    with redirect_stdout(stream):
        model.summary()
    summary_string = stream.getvalue()
    stream.close()
    return summary_string

# Assuming loss, model_file_name, epochs, and batch_size are already defined
model_details = {
    "model_summary": get_model_summary(model),
    "model_val_loss_final": loss,
    "model_file_name": model_file_name,
    "epochs": epochs,
    "batch_size": batch_size
}

print(model_details)

import json
import os
from datetime import datetime

# Save model details to a JSON file in the specified directory
output_path = r"desired path of save the model details"
json_file_path = os.path.join(output_path, f"{int(datetime.now().timestamp())}.json")
with open(json_file_path, "w") as f:
    json.dump(model_details, f)
    
    
    

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_path, f"loss_curves_epoch_{epochs}_{int(datetime.now().timestamp())}.png"))  # Save the plot
plt.show()  
