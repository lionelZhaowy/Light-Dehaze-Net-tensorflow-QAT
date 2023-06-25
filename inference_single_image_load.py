import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from LightdehazeNet_test import LightDehaze_Net
np.set_printoptions(threshold=np.inf)

saved_model_dir = './FP32_model/Epoch9'
tf.config.run_functions_eagerly(True)
# # Load the model
# model = tf.saved_model.load(saved_model_dir)
# 构建模型并加载预训练权重
annotated_model = LightDehaze_Net((3, 480, 640))  # 根据你的网络输入改变这里的shape

# Assume you have loaded a model named 'model'
model = tf.keras.models.load_model(saved_model_dir)

# Get weights of the model
weights = model.get_weights()

annotated_model.set_weights(weights)

# Use `quantize.apply` to actually make the model quantization aware.
quant_aware_model = tfmot.quantization.keras.quantize_apply(annotated_model)

# Load an image
image = Image.open('./Single_Image/NYU2_1.jpg')


if image is None:
    print("Failed to load image")
else:
    # Preprocess the image
    image = image.resize((640, 480))  # width and height should match the input size of your model
    with open("./log/image_int8.txt","w+") as f:
        f.write(str(image))
    image = np.array(image).astype(np.float32) / 255.0  # Convert to float32 and normalize
    image = np.expand_dims(image, axis=0)  # Add a batch dimension
# image = np.array(image).astype(np.float32)  # Convert to float32


image = np.transpose(image, (0, 3, 1, 2))  # if the model expects channels-first data

with open("./log/image_fp.txt","w+") as f:
    f.write(str(image))
# Perform inference
output = quant_aware_model(image)
with open("./log/result_fp.txt","w+") as f:
    f.write(str(output))
# # Reshape the output and remove batch dimension
# output_image = np.squeeze(output, axis=0)
#
# # Convert from channels-first to channels-last data if necessary
# output_image = np.transpose(output_image, (1, 2, 0))
#
# # Un-normalize the image
# output_image = output_image * 255
#
# # Convert to integer data type
# output_image = output_image.astype(np.uint8)
# with open("./log/result_int8.txt","w+") as f:
#     f.write(str(output_image))
# # Create a PIL image and save it
# output_image = Image.fromarray(output_image)
# output_image.save('output_image.jpg')
