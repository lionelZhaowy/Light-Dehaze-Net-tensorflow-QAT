import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from LightdehazeNet_test_INT8 import LightDehaze_Net
np.set_printoptions(threshold=np.inf)

saved_model_dir = './INT8_model/Epoch0'
tf.config.run_functions_eagerly(True)

# 构建模型并加载预训练权重
annotated_model = LightDehaze_Net((3, 480, 640))  # 根据你的网络输入改变这里的shape

# Assume you have loaded a model named 'model'
model = tf.keras.models.load_model(saved_model_dir)

# Get weights of the model
weights = model.get_weights()

# Use `quantize.apply` to actually make the model quantization aware.
quant_aware_model = tfmot.quantization.keras.quantize_apply(annotated_model)

quant_aware_model.set_weights(weights)

# Load an image
image = Image.open('./Single_Image/NYU2_1.jpg')


if image is None:
    print("Failed to load image")
else:
    # Preprocess the image
    image = image.resize((640, 480))  # width and height should match the input size of your model
    with open("./log/INT8/image_int8.txt","w+") as f:
        f.write(str(image))
    image = np.array(image).astype(np.float32) / 255.0  # Convert to float32 and normalize
    image = np.expand_dims(image, axis=0)  # Add a batch dimension
# image = np.array(image).astype(np.float32)  # Convert to float32


image = np.transpose(image, (0, 3, 1, 2))  # if the model expects channels-first data

with open("./log/INT8/image_fp.txt","w+") as f:
    f.write(str(image))

# Perform inference
outputs = quant_aware_model(image)
# Assuming that "outputs" is the list of outputs
for i, output in enumerate(outputs):
    with open(f"./log/INT8/QAT_{i}_INT8.txt", "w+") as f:
        f.write(str(output))

