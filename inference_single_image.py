import numpy as np
from PIL import Image
import tensorflow as tf

saved_model_dir = '.\\INT8_model\\Epoch0'

# Load the model
model = tf.saved_model.load(saved_model_dir)

# Load an image
image = Image.open('./Single_Image/NYU2_2.jpg')


if image is None:
    print("Failed to load image")
else:
    # Preprocess the image
    image = image.resize((640, 480))  # width and height should match the input size of your model
    image = np.array(image).astype(np.float32) / 255.0  # Convert to float32 and normalize
    image = np.expand_dims(image, axis=0)  # Add a batch dimension
# image = np.array(image).astype(np.float32)  # Convert to float32


image = np.transpose(image, (0, 3, 1, 2))  # if the model expects channels-first data

# Perform inference
output = model(image)

# Reshape the output and remove batch dimension
output_image = np.squeeze(output, axis=0)

# Convert from channels-first to channels-last data if necessary
output_image = np.transpose(output_image, (1, 2, 0))

# Un-normalize the image
output_image = output_image * 255

# Convert to integer data type
output_image = output_image.astype(np.uint8)

# Create a PIL image and save it
output_image = Image.fromarray(output_image)
output_image.save('output_image.jpg')
