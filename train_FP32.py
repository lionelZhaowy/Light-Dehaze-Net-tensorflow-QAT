from LightdehazeNet_FP32 import LightDehaze_Net
import image_data_loader
import tensorflow as tf
import os
from tqdm import tqdm
import tensorflow_model_optimization as tfmot

training_data_path = './data/original_images/images/'
validation_data_path = './data/training_images/data/'
FP32_model_path = './FP32_model/'
log_file_path = './FP32_model/log/'
num_of_epochs = 20

# Annotate model for quantization
annotated_model = LightDehaze_Net((3, 480, 640))


# 构建训练数据和评估数据加载器
training_data = image_data_loader.HazyDataLoader(training_data_path, validation_data_path)
validation_data = image_data_loader.HazyDataLoader(training_data_path, validation_data_path, mode="val")

# 使用 tf.data.Dataset.from_generator 从数据加载器创建数据集
train_dataset = tf.data.Dataset.from_generator(
    lambda: training_data,
    output_signature=(tf.TensorSpec(shape=(3, 480, 640), dtype=tf.float32),
                      tf.TensorSpec(shape=(3, 480, 640), dtype=tf.float32))
).batch(8) # 批次大小为 8

val_dataset = tf.data.Dataset.from_generator(
    lambda: validation_data,
    output_signature=(tf.TensorSpec(shape=(3, 480, 640), dtype=tf.float32),
                      tf.TensorSpec(shape=(3, 480, 640), dtype=tf.float32))
).batch(8) # 批次大小为 8

summary_writer = tf.summary.create_file_writer('./')


# 创建优化器和损失函数
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, decay=0.0001, clipnorm=0.1)
loss_fn = tf.keras.losses.MeanSquaredError()


# 在模型编译之后需要重新构建模型以使更改生效
annotated_model.compile(optimizer=optimizer, loss=loss_fn)
annotated_model.build(input_shape=(None, 3, 480, 640))

# 训练模型
min_loss = float('inf')  # 最小损失初始化
best_model_path = os.path.join(FP32_model_path, "best_model")
for epoch in range(num_of_epochs):
    # 对于每个epoch创建一个新的txt文件
    output_file = open(os.path.join(log_file_path, "output_epoch_" + str(epoch) + ".txt"), "w")
    print("Epoch:", epoch, file=output_file)
    print("Epoch:", epoch)
    avg_loss = 0
    iteration = 0
    for (hazefree_image, hazy_image) in tqdm(train_dataset):
        iteration += 1
        with tf.GradientTape() as tape:
            # 前向传播
            dehaze_image = annotated_model(hazy_image, training=True)
            loss = loss_fn(hazefree_image, dehaze_image)
        # 计算梯度
        gradients = tape.gradient(loss, annotated_model.trainable_variables)
        # 使用优化器更新模型参数
        optimizer.apply_gradients(zip(gradients, annotated_model.trainable_variables))

        avg_loss += loss

        if ((iteration+1) % 10) == 0:
            print("Loss at iteration", iteration+1, ":", loss.numpy(), file=output_file)
            print("Loss at iteration", iteration + 1, ":", loss.numpy())
        if ((iteration+1) % 200) == 0:
            # 保存模型
            avgg_loss = avg_loss / iteration
            print("Average loss now", epoch, ":", avgg_loss.numpy(), file=output_file)
            tf.keras.models.save_model(annotated_model, os.path.join(FP32_model_path, "Epoch" + str(epoch)))
    avg_loss /= iteration
    print("Average loss at epoch", epoch, ":", avg_loss.numpy(), file=output_file)
    print("Average loss at epoch", epoch, ":", avg_loss.numpy())

    # Validation Stage
    val_loss = 0
    val_iteration = 0
    for (val_hazefree_image, val_hazy_image) in tqdm(val_dataset):
        val_iteration += 1
        val_dehaze_image = annotated_model(val_hazy_image, training=False)
        val_loss += loss_fn(val_hazefree_image, val_dehaze_image)
    val_loss /= val_iteration

    if val_loss < min_loss:
        min_loss = val_loss
        tf.keras.models.save_model(annotated_model, best_model_path)
        print("Best model updated", file=output_file)
        print("Best model updated")
    print("Validation loss at epoch", epoch, ":", val_loss.numpy(), file=output_file)
    print("Validation loss at epoch", epoch, ":", val_loss.numpy())

    # 关闭当前epoch的txt文件
    output_file.close()

# 保存训练结束后的模型
tf.keras.models.save_model(annotated_model, os.path.join(FP32_model_path, "final_model"))
