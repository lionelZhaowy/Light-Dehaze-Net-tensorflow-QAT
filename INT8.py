import tensorflow as tf
import image_data_loader

# Load a SavedModel on disk.
saved_model_dir = '.\\FP32_model\\best_model'

training_data_path = './data/original_images/images/'
validation_data_path = './data/training_images/data/'

# 构建训练数据和评估数据加载器
training_data = image_data_loader.HazyDataLoader(training_data_path, validation_data_path)
validation_data = image_data_loader.HazyDataLoader(training_data_path, validation_data_path, mode="val")

# 使用 tf.data.Dataset.from_generator 从数据加载器创建数据集
train_dataset = tf.data.Dataset.from_generator(
    lambda: training_data,
    output_signature=(tf.TensorSpec(shape=(3, 480, 640), dtype=tf.float32),
                      tf.TensorSpec(shape=(3, 480, 640), dtype=tf.float32))
).batch(8) # 批次大小为 8


def representative_data_gen():
    for _ in range(100):
        # Get sample input data as a numpy array in a method of your choosing.
        hazefree_image, hazy_image = next(iter(train_dataset))
        yield [hazy_image]


# Convert the model.
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen  # function to generate representative dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_quant_model = converter.convert()

# 保存量化后的模型
with open('lightdehaze_model.tflite', 'wb') as f:
  f.write(tflite_quant_model)