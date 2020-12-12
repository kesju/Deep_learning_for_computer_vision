# plot dog photos from the dogs vs cats dataset
from matplotlib import pyplot
from matplotlib.image import imread
# define location of dataset
folder = 'C:/DI/Data/data/train/rasa/'
# plot first few images

for i in range(7):
	# define subplot
	pyplot.subplot(330 + 1 + i)
	# define filename
	filename = folder + 'rasa000' + str(i) + '.jpg'
	print(filename)
	# load image pixels
	image = imread(filename)
	# plot raw pixel data
	pyplot.imshow(image)
# show the figure
pyplot.show()

# define location of dataset
folder = 'C:/DI/Data/data/train/egle/'
# plot first few images

for i in range(7):
	# define subplot
	pyplot.subplot(330 + 1 + i)
	# define filename
	filename = folder + 'egle000' + str(i) + '.jpg'
	print(filename)
	# load image pixels
	image = imread(filename)
	# plot raw pixel data
	pyplot.imshow(image)
# show the figure
pyplot.show()
