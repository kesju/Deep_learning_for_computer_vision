# 1-vgg block baseline model for the dogs vs cats dataset
import sys
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd


def run_test_harness():
    datagen = ImageDataGenerator(rescale=1.0/255.0)
    # prepare iterators
    test_it = datagen.flow_from_directory('C:/DI/Data/dataset_dog_cat/test/',
                                          class_mode='binary', batch_size=16, target_size=(200, 200))
    print(test_it)
    print("test_it length =", len(test_it))
    print("test_it batch size =", test_it.batch_size)

    # for file in test_it.filenames:
    # print(file)
    # for label in test_it.labels:
    # print(label)

# List Comprehension
    s1 = [file for file in test_it.filenames]
    s2 = [label for label in test_it.labels]

    df = pd.DataFrame()
    df['Path'] = s1
    df['Labels'] = s2
    print(df)

    print(predictions)
    b_labels = (model.predict(train_it) > 0.5).astype(int)
    print("b_labels =", b_labels)
    class_names = ['Cat', 'Dog']
     # print(class_names[np.argmax(b_labels)])
    print(b_labels.shape)
    assigned = []
    for i in range(len(b_labels)):
        index = b_labels[i, 0]
        print(index)
        assigned.append(class_names[index])
    print("Assigned = ", assigned)


# entry point, run the test harness
run_test_harness()
