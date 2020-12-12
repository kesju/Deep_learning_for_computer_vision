import numpy as np
import tensorflow as tf



def conv_2d(x, kernel, bias):
    kernel_shape = kernel.shape[0]
    
    # Assuming Padding = 0, stride = 1
    output_shape = x.shape[0] - kernel_shape + 1
    result = np.zeros((output_shape, output_shape))
    
    for row in range(x.shape[0] - 1):
        for col in range(x.shape[1] - 1):
            window = x[row: row + kernel_shape, col: col + kernel_shape]
            result[row, col] = np.sum(np.multiply(kernel,window))
    return result + bias


input_matrix = np.array([[3., 9., 0.],[2., 8., 1.],[1., 4., 8.]])
kernel = np.array([[8., 9.],[4., 4.]])
bias = np.array([0.06])


naive_conv_op = conv_2d(input_matrix, kernel, bias)
print(naive_conv_op)

# torch.nn.Conv2d(in_channels: int, out_channels: int, kernel_size: Union)
# torch_conv = nn.Conv2d(1, 1, 2)
# torch_conv_op = torch_conv(input_matrix)
# print(torch_conv_op)

"""
# Computes a 2-D convolution given input and 4-D filters tensors.
# tf.nn.conv2d(
    input, filters, strides, padding, data_format='NHWC', dilations=None, name=None
)
https://www.tensorflow.org/api_docs/python/tf/nn/conv2d
"""
tf_conv_op = tf.nn.conv2d()