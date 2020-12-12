# plot dog photos from the dogs vs cats dataset
from matplotlib import pyplot
from matplotlib.image import imread
# define location of dataset
folder = 'C:/DI/Data/PetImages/Dog/'
# plot first few images

for i in range(9):
	# define subplot
	pyplot.subplot(330 + 1 + i)
	# define filename
	filename = folder + str(i) + '.jpg'
	print(filename)
	# load image pixels
	image = imread(filename)
	# plot raw pixel data
	pyplot.imshow(image)
# show the figure
pyplot.show()

# define location of dataset
folder = 'C:/DI/Data/PetImages/Cat/'
# plot first few images

for i in range(7):
	# define subplot
	pyplot.subplot(330 + 1 + i)
	# define filename
	filename = folder + str(i) + '.jpg'
	print(filename)
	# load image pixels
	image = imread(filename)
	# plot raw pixel data
	pyplot.imshow(image)
# show the figure
pyplot.show()
