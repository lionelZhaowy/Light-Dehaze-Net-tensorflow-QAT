# Author：xiaodiao
# Created Date: 2023-06-17

import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np
from keras import backend as K


def get_numpy_array(model, my_tensor,input_data):
    # Define a TensorFlow function that takes inputs and outputs the Keras tensor value
    input_tensors = model.inputs + [K.learning_phase()]
    output_tensors = [my_tensor]
    k_function = K.function(input_tensors, output_tensors)

    # Call the TensorFlow function with the input data to get the tensor value(s)
    tensor_value = k_function([input_data, 0])[0]

    # Convert the tensor value to a NumPy array and return it
    np_array = np.array(tensor_value)
    return np_array


def LightDehaze_Net(input_shape):
    tf.config.run_functions_eagerly(True)

    # Define the input
    img = tf.keras.Input(shape=input_shape)

    # LightDehazeNet Architecture
    relu = tfmot.quantization.keras.quantize_annotate_layer(tf.keras.layers.ReLU())

    e_conv_layer1 = tfmot.quantization.keras.quantize_annotate_layer(tf.keras.layers.Conv2D(3, 1, (1, 1), data_format='channels_first', padding='valid', use_bias=True))
    e_conv_layer2 = tfmot.quantization.keras.quantize_annotate_layer(tf.keras.layers.Conv2D(3, 3, (1, 1), data_format='channels_first', padding='same', use_bias=True))
    e_conv_layer3 = tfmot.quantization.keras.quantize_annotate_layer(tf.keras.layers.Conv2D(3, 5, (1, 1), data_format='channels_first', padding='same', use_bias=True))
    e_conv_layer4 = tfmot.quantization.keras.quantize_annotate_layer(tf.keras.layers.Conv2D(6, 7, (1, 1), data_format='channels_first', padding='same', use_bias=True))
    e_conv_layer5 = tfmot.quantization.keras.quantize_annotate_layer(tf.keras.layers.Conv2D(6, 3, (1, 1), data_format='channels_first', padding='same', use_bias=True))
    e_conv_layer6 = tfmot.quantization.keras.quantize_annotate_layer(tf.keras.layers.Conv2D(6, 3, (1, 1), data_format='channels_first', padding='same', use_bias=True))
    e_conv_layer7 = tfmot.quantization.keras.quantize_annotate_layer(tf.keras.layers.Conv2D(6, 3, (1, 1), data_format='channels_first', padding='same', use_bias=True))
    e_conv_layer8 = tfmot.quantization.keras.quantize_annotate_layer(tf.keras.layers.Conv2D(3, 3, (1, 1), data_format='channels_first', padding='same', use_bias=True))

    # Apply annotation in the call function
    conv_layer1 = e_conv_layer1(img)

    conv_layer1 = relu(conv_layer1)

    conv_layer2 = e_conv_layer2(conv_layer1)

    conv_layer2 = relu(conv_layer2)

    conv_layer3 = e_conv_layer3(conv_layer2)

    conv_layer3 = relu(conv_layer3)

    # concatenating conv1 and conv3
    concat_layer1 = tf.concat((conv_layer1, conv_layer3), axis=1)

    conv_layer4 = e_conv_layer4(concat_layer1)

    conv_layer4 = relu(conv_layer4)

    conv_layer5 = e_conv_layer5(conv_layer4)

    conv_layer5 = relu(conv_layer5)

    conv_layer6 = relu(e_conv_layer6(conv_layer5))

    conv_layer6 = relu(conv_layer6)

    # concatenating conv4 and conv6
    concat_layer2 = tf.concat((conv_layer4, conv_layer6), axis=1)

    conv_layer7 = relu(e_conv_layer7(concat_layer2))

    conv_layer7 = relu(conv_layer7)

    # concatenating conv2, conv5, and conv7
    concat_layer3 = tf.concat((conv_layer2, conv_layer5, conv_layer7), axis=1)

    conv_layer8 = e_conv_layer8(concat_layer3)

    conv_layer8 = relu(conv_layer8)

    dehaze_image = (conv_layer8 * img)

    dehaze_image = dehaze_image - conv_layer8

    dehaze_image = dehaze_image + 1

    dehaze_image = relu(dehaze_image)


    # J(x) = clean_image, k(x) = x8, I(x) = x, b = 1

    # Construct the model
    model = tf.keras.Model(inputs=img, outputs=[conv_layer1, conv_layer2, conv_layer3, concat_layer1, conv_layer4, conv_layer5, conv_layer6, concat_layer2, conv_layer7, concat_layer3, conv_layer8, dehaze_image])


    return model



# def LightDehaze_Net(input_shape):
#     # Define the input
#     img = tf.keras.Input(shape=input_shape)

#     # LightDehazeNet Architecture
#     relu = tfmot.quantization.keras.quantize_annotate_layer(tf.keras.layers.ReLU())

#     e_conv_layer1 = tfmot.quantization.keras.quantize_annotate_layer(tf.keras.layers.Conv2D(3, 1, (1, 1), data_format='channels_first', padding='valid', use_bias=True))
#     e_conv_layer2 = tfmot.quantization.keras.quantize_annotate_layer(tf.keras.layers.Conv2D(3, 3, (1, 1), data_format='channels_first', padding='same', use_bias=True))
#     e_conv_layer3 = tfmot.quantization.keras.quantize_annotate_layer(tf.keras.layers.Conv2D(3, 5, (1, 1), data_format='channels_first', padding='same', use_bias=True))
#     e_conv_layer4 = tfmot.quantization.keras.quantize_annotate_layer(tf.keras.layers.Conv2D(6, 7, (1, 1), data_format='channels_first', padding='same', use_bias=True))
#     e_conv_layer5 = tfmot.quantization.keras.quantize_annotate_layer(tf.keras.layers.Conv2D(6, 3, (1, 1), data_format='channels_first', padding='same', use_bias=True))
#     e_conv_layer6 = tfmot.quantization.keras.quantize_annotate_layer(tf.keras.layers.Conv2D(6, 3, (1, 1), data_format='channels_first', padding='same', use_bias=True))
#     e_conv_layer7 = tfmot.quantization.keras.quantize_annotate_layer(tf.keras.layers.Conv2D(6, 3, (1, 1), data_format='channels_first', padding='same', use_bias=True))
#     e_conv_layer8 = tfmot.quantization.keras.quantize_annotate_layer(tf.keras.layers.Conv2D(3, 3, (1, 1), data_format='channels_first', padding='same', use_bias=True))

#     # Apply annotation in the call function
#     conv_layer1 = relu(e_conv_layer1(img))
#     conv_layer2 = relu(e_conv_layer2(conv_layer1))
#     conv_layer3 = relu(e_conv_layer3(conv_layer2))

#     # concatenating conv1 and conv3
#     concat_layer1 = tf.concat((conv_layer1, conv_layer3), axis=1)

#     conv_layer4 = relu(e_conv_layer4(concat_layer1))
#     conv_layer5 = relu(e_conv_layer5(conv_layer4))
#     conv_layer6 = relu(e_conv_layer6(conv_layer5))

#     # concatenating conv4 and conv6
#     concat_layer2 = tf.concat((conv_layer4, conv_layer6), axis=1)

#     conv_layer7 = relu(e_conv_layer7(concat_layer2))

#     # concatenating conv2, conv5, and conv7
#     concat_layer3 = tf.concat((conv_layer2, conv_layer5, conv_layer7), axis=1)

#     conv_layer8 = relu(e_conv_layer8(concat_layer3))

#     dehaze_image = relu((conv_layer8 * img) - conv_layer8 + 1)

#     # J(x) = clean_image, k(x) = x8, I(x) = x, b = 1

#     # Construct the model
#     model = tf.keras.Model(inputs=img, outputs=dehaze_image)

#     return model

