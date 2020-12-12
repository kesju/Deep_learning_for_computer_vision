import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np
print("tf versija ", tf.__version__)

"""
Šioje programoje trejais būdais apskaičiuojama konvoliucija
(iš tikro, tai kroskovariacija):
1) naudojant ciklus ir indeksus - naive computation,
tam sukurta funkcija conv_2d.
Šaltinis: How Are Convolutions Actually Performed Under the Hood?
Anirudh Shenoy, 2019
From <https://towardsdatascience.com/how-are-convolutions-actually
-performed-under-the-hood-226523ce7fbf>
Iš šio šaltinio paimti duomenys ir panaudoti visiems variantams.

2) naudojant tensorflow funkciją tf.nn.conv2d.
Šaltinis: tf.nn.conv2d aprašymas. Computes a 2-D convolution given
input and 4-D filters tensors.
# tf.nn.conv2d(input, filters, strides, padding, data_format='NHWC',
    dilations=None, name=None)
https://www.tensorflow.org/api_docs/python/tf/nn/conv2d

3) naudojant keras funkciją tf.conv2d. 
Šaltinis: knyga Deep Learning for Computer Vision, Jason Brownlee, 2019 
From <https://machinelearningmastery.com/deep-learning-for-computer-vision/> 
Chapter 11
"""
# -------------------------------------------------------------------------
# 1-as būdas: naive computation

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

input_matrix = np.array([
                        [3., 9., 0.],
                        [2., 8., 1.],
                        [1., 4., 8.]
                        ])
kernel = np.array([
                  [8., 9.],
                  [4., 4.]
                  ])

bias = np.array([0.00])

naive_conv_op = conv_2d(input_matrix, kernel, bias)
print("\nI-as variantas: paprastesnis")
print("\n1-as būdas: naive computation")
print('-----------------------------------------------------------------')
print("\ninput_matrix.shape = ", input_matrix.shape)
print("\nkernel.shape = ",kernel.shape)
print('\ninput_matrix = ',input_matrix)
print('\nkernel = ',kernel)
print("\nresult = ", naive_conv_op)
print("\n\n")
# -------------------------------------------------------------------------
# 2-as būdas: naudojant tensorflow funkciją tf.nn.conv2d

# Pastaba: duomenys pakartojau 2 kartus (batch = 2),
# galima kartoti kiek nori, bus tas pats rezultatas
x_in = np.array([
    [
    [[3.], [9.], [0.]],
    [[2.], [8.], [1.]],
    [[1.], [4.], [8.]],
    ],
    [
    [[3.], [9.], [0.]],
    [[2.], [8.], [1.]],
    [[1.], [4.], [8.]],]
    ])
# x_in shape: [2,3,3,1]
# (batch_shape + [in_height, in_width, in_channels])

kernel_in = np.array(
 [
    [[[8.]],
    [[9.]]],
    [[[4.]],
    [[4.]]]
 ])
# kernel_in shape: [2,2,1,1]
# [filter_height, filter_width, in_channels, out_channels]

print("\n\n2-as būdas: naudojant tensorflow funkciją tf.nn.conv2d")
print('--------------------------------------------')

print("\nx_in.shape = ", x_in.shape)
print("\nkernel_in.shape = ",kernel_in.shape)

x = tf.constant(x_in, dtype=tf.float32)
print('\nx = ',x)
kernel = tf.constant(kernel_in, dtype=tf.float32)
print("\nkernel = ",kernel)
c = tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='VALID')
print("\nresult = ", c)
print("\n\n")

# -------------------------------------------------------------------------
# 3-ias būdas: naudojant keras funkciją tf.conv2d
# example of calculation 2d convolutions
from numpy import asarray
from keras.models import Sequential
from keras.layers import Conv2D
# define input data
# Duomenys tokie patys, kaip ir 2 variante.

# create model
model = Sequential()
model.add(Conv2D(1, (2,2), input_shape=(3, 3, 1)))
weights = [asarray(kernel_in), asarray([0.0])]
print("\n\n3-ias būdas: naudojant keras funkciją tf.conv2d")
print('--------------------------------------------')
print("\n weights =", weights)
# store the weights in the model
# model.set_weights(weights)
model.set_weights(weights)
# confirm they were stored
print(model.get_weights())
# apply filter to input data
yhat = model.predict(x_in)
for r in range(yhat.shape[1]):
	# print each column in the row
	print([yhat[0,r,c,0] for c in range(yhat.shape[2])])


"""
# II-as sudėtingesnis variantas 
# -------------------------------------------------------------------------
# 1-as būdas: naive computation


input_matrix = np.array([
                        [3., 9., 0.],
                        [2., 8., 1.],
                        [1., 4., 8.]
                        ])
kernel = np.array([
                  [8., 9.],
                  [4., 4.]
                  ])

bias = np.array([0.00])

naive_conv_op = conv_2d(input_matrix, kernel, bias)
print("\nII-as variantas: sudėtingesnis")
print("1-as būdas: naive computation")
print('-----------------------------------------------------------------')
print("\ninput_matrix.shape = ", input_matrix.shape)
print("\nkernel.shape = ",kernel.shape)
print('\ninput_matrix = ',input_matrix)
print('\nkernel = ',kernel)
print("\nresult = ", naive_conv_op)
print("\n\n")
# -------------------------------------------------------------------------
# 2-as būdas: naudojant tensorflow funkciją tf.nn.conv2d

# Pastaba: duomenys pakartojau 2 kartus (batch = 2),
# galima kartoti kiek nori, bus tas pats rezultatas
x_in = np.array([
    [
    [[3.], [9.], [0.]],
    [[2.], [8.], [1.]],
    [[1.], [4.], [8.]],
    ],
    [
    [[3.], [9.], [0.]],
    [[2.], [8.], [1.]],
    [[1.], [4.], [8.]],]
    ])
# x_in shape: [2,3,3,1]
# (batch_shape + [in_height, in_width, in_channels])

kernel_in = np.array(
 [
    [[[8.]],
    [[9.]]],
    [[[4.]],
    [[4.]]]
 ])
# kernel_in shape: [2,2,1,1]
# [filter_height, filter_width, in_channels, out_channels]

print("\n\n2-as būdas: naudojant tensorflow funkciją tf.nn.conv2d")
print('--------------------------------------------')

print("\nx_in.shape = ", x_in.shape)
print("\nkernel_in.shape = ",kernel_in.shape)

x = tf.constant(x_in, dtype=tf.float32)
print('\nx = ',x)
kernel = tf.constant(kernel_in, dtype=tf.float32)
print("\nkernel = ",kernel)
c = tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='VALID')
print("\nresult = ", c)
print("\n\n")

# -------------------------------------------------------------------------
# 3-ias būdas: naudojant keras funkciją tf.conv2d
# example of calculation 2d convolutions
from numpy import asarray
from keras.models import Sequential
from keras.layers import Conv2D
# define input data
# Duomenys tokie patys, kaip ir 2 variante.

# create model
model = Sequential()
model.add(Conv2D(1, (2,2), input_shape=(3, 3, 1)))
weights = [asarray(kernel_in), asarray([0.0])]
print("\n\n3-ias būdas: naudojant keras funkciją tf.conv2d")
print('--------------------------------------------')
print("\n weights =", weights)
# store the weights in the model
# model.set_weights(weights)
model.set_weights(weights)
# confirm they were stored
print(model.get_weights())
# apply filter to input data
yhat = model.predict(x_in)
for r in range(yhat.shape[1]):
	# print each column in the row
	print([yhat[0,r,c,0] for c in range(yhat.shape[2])])
"""