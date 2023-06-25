# Author：xiaodiao
# Created Date: 2023-06-17

import tensorflow as tf
import tensorflow_model_optimization as tfmot

def LightDehaze_Net(input_shape):
    # Define the input
    img = tf.keras.Input(shape=input_shape)

    # LightDehazeNet Architecture
    relu = tf.keras.layers.ReLU()

    e_conv_layer1 = tf.keras.layers.Conv2D(3, 1, (1, 1), data_format='channels_first', padding='valid', use_bias=True)
    e_conv_layer2 = tf.keras.layers.Conv2D(3, 3, (1, 1), data_format='channels_first', padding='same', use_bias=True)
    e_conv_layer3 = tf.keras.layers.Conv2D(3, 5, (1, 1), data_format='channels_first', padding='same', use_bias=True)
    e_conv_layer4 = tf.keras.layers.Conv2D(6, 7, (1, 1), data_format='channels_first', padding='same', use_bias=True)
    e_conv_layer5 = tf.keras.layers.Conv2D(6, 3, (1, 1), data_format='channels_first', padding='same', use_bias=True)
    e_conv_layer6 = tf.keras.layers.Conv2D(6, 3, (1, 1), data_format='channels_first', padding='same', use_bias=True)
    e_conv_layer7 = tf.keras.layers.Conv2D(6, 3, (1, 1), data_format='channels_first', padding='same', use_bias=True)
    e_conv_layer8 = tf.keras.layers.Conv2D(3, 3, (1, 1), data_format='channels_first', padding='same', use_bias=True)

    # Apply annotation in the call function
    conv_layer1 = relu(e_conv_layer1(img))
    conv_layer2 = relu(e_conv_layer2(conv_layer1))
    conv_layer3 = relu(e_conv_layer3(conv_layer2))

    # concatenating conv1 and conv3
    concat_layer1 = tf.concat((conv_layer1, conv_layer3), axis=1)

    conv_layer4 = relu(e_conv_layer4(concat_layer1))
    conv_layer5 = relu(e_conv_layer5(conv_layer4))
    conv_layer6 = relu(e_conv_layer6(conv_layer5))

    # concatenating conv4 and conv6
    concat_layer2 = tf.concat((conv_layer4, conv_layer6), axis=1)

    conv_layer7 = relu(e_conv_layer7(concat_layer2))

    # concatenating conv2, conv5, and conv7
    concat_layer3 = tf.concat((conv_layer2, conv_layer5, conv_layer7), axis=1)

    conv_layer8 = relu(e_conv_layer8(concat_layer3))

    dehaze_image = relu((conv_layer8 * img) - conv_layer8 + 1)
    # J(x) = clean_image, k(x) = x8, I(x) = x, b = 1

    # Construct the model
    model = tf.keras.Model(inputs=img, outputs=dehaze_image)

    return model

