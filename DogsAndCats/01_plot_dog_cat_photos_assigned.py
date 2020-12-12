# plot dog photos from the dogs vs cats dataset
from matplotlib import pyplot
from matplotlib.image import imread
import pandas as pd

# define location of assigned labels
folder_df = 'C:\GitHub\Deep_learning_for_computer_vision\Mano\dog_cat_assigned_pretrained.csv'
df = pd.read_csv(folder_df)

# define location of image dataset
folder_dataset = 'C:\DI\Data\PetImages'
# which image to plot
idx = 7
file,label,assigned = df.iloc[idx,1:4]
filename = folder_dataset + '\\' + file
print(file,label,assigned)
# load image pixels
image = imread(filename)
# plot raw pixel data
image_title = str(idx) + '  ' + file + '  ' + label + '  Assigned: ' + assigned
pyplot.title(image_title)
pyplot.imshow(image)
# show the figure
pyplot.show()
