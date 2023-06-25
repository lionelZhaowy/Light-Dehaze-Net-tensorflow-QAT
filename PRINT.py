import tensorflow as tf
import numpy as np
import os
np.set_printoptions(threshold=np.inf)

# 创建weights文件夹
if not os.path.exists('weights'):
    os.makedirs('weights')

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="lightdehaze_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Loop over all the operators in the model and extract the weights.
for op in interpreter.get_tensor_details():
    if op['name'] not in [i['name'] for i in input_details + output_details]:
        weight = interpreter.get_tensor(op['index'])
        scale = op['quantization_parameters']['scales']
        zero_point = op['quantization_parameters']['zero_points']

        # 将操作的名字中的"/"替换为"_"，以避免在文件系统中创建子目录
        op_name = op['name'].replace('/', '_')

        # 打开文件，准备写入
        with open(f'weights/{op_name}.txt', 'w') as file:
            # 写入权重的形状
            file.write(f"Shape: {np.array(weight).shape}\n")
            # scale
            file.write(f"Scale: {scale}\n")
            # zero point
            file.write(f"Zero_point: {zero_point}\n")
            # 写入权重的数据类型
            file.write(f"Type: {weight.dtype}\n")

            # 对于0维的权重（即标量），直接写入
            if weight.ndim == 0:
                file.write(str(weight))
            else:
                # 将权重保存为二维数组
                if weight.ndim > 2:
                    weight = weight.reshape((weight.shape[0], -1))

                # 写入权重值
                np.savetxt(file, weight, delimiter=' ', fmt='%s', newline='\n', footer='', comments='', encoding=None)

        print(f"Operator: {op_name}, Weights shape: {np.array(weight).shape}, Type: {weight.dtype}")

