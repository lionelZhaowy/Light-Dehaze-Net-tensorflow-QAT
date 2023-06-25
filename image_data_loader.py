# Author：xiaodiao
# Created Date: 2023-06-17

import tensorflow as tf
import numpy as np
from PIL import Image
import glob
import random

random.seed(1143)

# 准备训练数据
def preparing_training_data(hazefree_images_dir, hazeeffected_images_dir):
    print('loading')
    train_data = []  # 存储训练数据的列表
    validation_data = []  # 存储验证数据的列表

    hazy_data = glob.glob(hazeeffected_images_dir + "*.jpg")  # 获取所有模糊图像的文件路径

    data_holder = {}  # 用于按照ID存储模糊图像

    # 遍历所有模糊图像，根据ID将其存储到data_holder中
    for h_image in hazy_data:
        h_image = h_image.split("\\")[-1]

        id_ = h_image.split("_")[0] + "_" + h_image.split("_")[1] + ".jpg"

        if id_ in data_holder.keys():
            data_holder[id_].append(h_image)
        else:
            data_holder[id_] = []
            data_holder[id_].append(h_image)

    train_ids = []  # 存储训练数据ID的列表
    val_ids = []    # 存储验证数据ID的列表

    num_of_ids = len(data_holder.keys())  # ID的总数
    for i in range(num_of_ids):
        if i < num_of_ids * 9 / 10:
            train_ids.append(list(data_holder.keys())[i])  # 将90%的ID添加到训练数据ID列表中
        else:
            val_ids.append(list(data_holder.keys())[i])    # 将10%的ID添加到验证数据ID列表中

    # 根据ID和图像文件名构建训练数据和验证数据的列表
    for id_ in list(data_holder.keys()):
        if id_ in train_ids:
            for hazy_image in data_holder[id_]:
                train_data.append([hazefree_images_dir + id_, hazeeffected_images_dir + hazy_image])
        else:
            for hazy_image in data_holder[id_]:
                validation_data.append([hazefree_images_dir + id_, hazeeffected_images_dir + hazy_image])

    random.shuffle(train_data)      # 随机打乱训练数据顺序
    random.shuffle(validation_data) # 随机打乱验证数据顺序

    return train_data, validation_data

# 自定义数据加载器
class HazyDataLoader(tf.keras.utils.Sequence):
    def __init__(self, hazefree_images_dir, hazeeffected_images_dir, mode='train'):
        self.train_data, self.validation_data = preparing_training_data('./data/original_images/images/', './data/training_images/data/')

        if mode == 'train':
            self.data_dict = self.train_data
            print("Number of Training Images:", len(self.train_data))
        else:
            self.data_dict = self.validation_data
            print("Number of Validation Images:", len(self.validation_data))

    def __getitem__(self, index):
        hazefree_image_path, hazy_image_path = self.data_dict[index]

        print(hazefree_image_path, hazy_image_path)

        hazefree_image = Image.open(hazefree_image_path)  # 打开无雾图像
        hazy_image = Image.open(hazy_image_path)          # 打开有雾图像

        hazefree_image = hazefree_image.resize((640, 480), Image.ANTIALIAS)  # 调整无雾图像的大小
        hazy_image = hazy_image.resize((640, 480), Image.ANTIALIAS)          # 调整有雾图像的大小

        hazefree_image = (np.asarray(hazefree_image) / 255.0)   # 将无雾图像转换为数组并进行归一化
        hazy_image = (np.asarray(hazy_image) / 255.0)           # 将有雾图像转换为数组并进行归一化

        hazefree_image = tf.convert_to_tensor(hazefree_image, dtype=tf.float32) # 将无雾图像转换为TensorFlow张量
        hazy_image = tf.convert_to_tensor(hazy_image, dtype=tf.float32)         # 将有雾图像转换为TensorFlow张量

        return tf.transpose(hazefree_image, [2, 0, 1]), tf.transpose(hazy_image, [2, 0, 1])  # 转置张量的维度顺序

    def __len__(self):
        return len(self.data_dict)  # 返回数据集的长度
