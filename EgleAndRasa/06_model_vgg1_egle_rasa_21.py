# 1-vgg block baseline model for the dogs vs cats dataset
import sys
from matplotlib import pyplot
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

# define cnn model


def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu',
	          kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
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
    class_names = ['Egle', 'Rasa']
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
	datagen = ImageDataGenerator(rescale=1.0/255.0)
# prepare iterators
	train_it = datagen.flow_from_directory('C:/DI/Data/dataset_egle_rasa/train/',
		class_mode='binary', batch_size=6, target_size=(200, 200))
	print('length of train_it =', len(train_it))
	test_it = datagen.flow_from_directory('C:/DI/Data/dataset_egle_rasa/test/',
		class_mode='binary', batch_size=5, target_size=(200, 200))
	print('length of test_it =', len(test_it))

# fit model
	history = model.fit(train_it, steps_per_epoch=len(train_it),
		validation_data=test_it, validation_steps=len(test_it), epochs=1, verbose=1)
# evaluate model
	_, acc = model.evaluate(test_it, steps=len(test_it), verbose=1)
	print('> %.3f' % (acc * 100.0))
# learning curves
	summarize_diagnostics(history)
	predictions = model.predict(test_it)
	df = prediction_labels(model,test_it)
	df.to_csv('egle_rasa_vgg1.csv') 

# entry point, run the test harness
predictions = run_test_harness()
