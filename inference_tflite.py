import numpy as np
from PIL import Image
import tensorflow as tf

tflite_model_file = '.\\lightdehaze_model.tflite'


interpreter = tf.lite.Interpreter(model_path=tflite_model_file)
interpreter.allocate_tensors()

image = Image.open('./Single_Image/NYU2_2.jpg')

if image is None:
    print("Failed to load image")
else:
    # Preprocess the image
    image = image.resize((640, 480))
    image = np.array(image).astype(np.int8)
    image = np.expand_dims(image, axis=0)
    image = np.transpose(image, (0, 3, 1, 2))

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on input data
input_shape = input_details[0]['shape']
input_data = np.array(image, dtype=np.int8)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])

output_image = np.squeeze(output_data, axis=0)
output_image = np.transpose(output_image, (1, 2, 0))
output_image = output_image
output_image = output_image.astype(np.uint8)
output_image = Image.fromarray(output_image)
output_image.save('output_image.jpg')
