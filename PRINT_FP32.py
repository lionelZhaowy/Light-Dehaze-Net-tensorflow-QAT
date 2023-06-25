import tensorflow as tf
import numpy as np
# Set print options
np.set_printoptions(threshold=np.inf)

# 加载模型
loaded_model = tf.keras.models.load_model('.\INT8_model\Epoch0')

# 获取模型的所有层
layers = loaded_model.layers

with open('all_weights.txt', 'w') as f:
    for i, layer in enumerate(layers):
        weights = layer.get_weights()  # 返回一个numpy数组列表
        for j, weight in enumerate(weights):
            weight_info = "Layer {} Weight {}: {} with shape {}, type: {}".format(i, j, layer.weights[j].name, weight.shape, type(weight))
            f.write(weight_info + "\n")  # 写入参数信息

            if weight.ndim > 2:
                # 如果weight的维度大于2，我们将其展平为一维数组再保存
                weight = weight.flatten()
            elif weight.ndim == 0:
                # 如果weight的维度为0，我们将其变为一个一维数组再保存
                weight = np.array([weight])

            np.savetxt(f, weight)  # 保存参数值
            f.write("\n")  # 在不同参数之间添加一个空行，便于阅读



