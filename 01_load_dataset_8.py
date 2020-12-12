# example of progressively loading images from file
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img

def show(batchX):
    for i in range(batchX.shape[0]):
        image = batchX[i,:,:,:]
        image2 = array_to_img(image)
        print("converting NumPy array:",type(image2))
        image2.show()

# create generator
datagen = ImageDataGenerator()
# prepare an iterators for each dataset
train_it = datagen.flow_from_directory('c:/DI/Data/data/train/',
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=1,
    class_mode="binary",
    shuffle=False,
    seed=42)
# val_it = datagen.flow_from_directory('c:/DI/Data/data/validation/', class_mode='binary')
# test_it = datagen.flow_from_directory('c:/DI/Data/data/test/', class_mode='binary')
# confirm the iterator works
# batchX, batchy = train_it.next()

"""
done_looping = False
    while not done_looping:
        try:
            item = next(iterator)
        except StopIteration:
            done_looping = True
        else:
            action_to_do(item)
"""

done_looping = False
while not done_looping:
    try:
        batchX, batchy = train_it.next()
    except StopIteration:
        done_looping = True
    else:
        print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))
        print(batchy)

# for batchX, batchy in train_it:
    # print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))
    # print(batchy)
    # show(batchX)



"""
from matplotlib import pyplot
# display the array of pixels as an image
image = batchX[0,:,:,:]
image = image.astype(int)
print(image.dtype)
print(image.shape)
# pyplot.imshow(image)
# pyplot.show()
data = batchX[0,:,:,:]
print(data.dtype)
print(data.shape)

image2 = array_to_img(data)
print("converting NumPy array:",type(image2))
image2.show()

# data = data.astype(int)
# print(image2.dtype)
# print(image2.shape)
# image2 = Image.fromarray(data)
# print(type(image2))
# pyplot.imshow(image2)
image2.show()
"""