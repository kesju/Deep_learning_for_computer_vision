# vgg16 model used for transfer learning on the dogs and cats dataset
import sys
from matplotlib import pyplot
import pandas as pd
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

# define cnn model


def define_model():
    # load model
    model = VGG16(include_top=False, input_shape=(224, 224, 3))
    # mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False
    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation='relu',
                   kernel_initializer='he_uniform')(flat1)
    output = Dense(1, activation='sigmoid')(class1)
    # define new model
    model = Model(inputs=model.inputs, outputs=output)
    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# plot diagnostic learning curves


def summarize_diagnostics(history):
    # plot loss
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
    # save plot to file
    filename = sys.argv[0].split('/')[-1]
    pyplot.savefig(filename + '_plot.png')
    pyplot.close()


def prediction_labels(model, test_it):
    class_names = ['Cat', 'Dog']
    s1 = [file for file in test_it.filenames]
    s2 = [class_names[label] for label in test_it.labels]
    df = pd.DataFrame()
    df['Path'] = s1
    df['Labels'] = s2

    b_labels = (model.predict(test_it) > 0.5).astype(int)
    print(b_labels.shape)

    assigned = []
    for i in range(len(b_labels)):
        index = b_labels[i, 0]
        assigned.append(class_names[index])
    df['Assigned'] = assigned
    return df


# run the test harness for evaluating a model
def run_test_harness():
# define model
    model = define_model()
# create data generator
    datagen = ImageDataGenerator(featurewise_center=True)
# specify imagenet mean values for centering
    datagen.mean = [123.68, 116.779, 103.939]
# prepare iterator
    train_it = datagen.flow_from_directory('C:/DI/Data/dataset_dog_cat/train/',
                                           class_mode='binary', batch_size=3, target_size=(224, 224))
    test_it = datagen.flow_from_directory('C:/DI/Data/dataset_dog_cat/test/',
                                          class_mode='binary', batch_size=2, target_size=(224, 224))
# fit model
    history = model.fit(train_it, steps_per_epoch=len(train_it),
                        validation_data=test_it, validation_steps=len(test_it), epochs=1, verbose=1)
# evaluate model
    _, acc = model.evaluate(test_it, steps=len(test_it), verbose=0)
    print('> %.3f' % (acc * 100.0))
# learning curves
    summarize_diagnostics(history)
    predictions = model.predict(test_it)
    df = prediction_labels(model, test_it)
    df.to_csv('dog_cat_assigned_pretrained.csv')


# entry point, run the test harness
run_test_harness()
