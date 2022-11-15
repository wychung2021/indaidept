import matplotlib.pyplot as plt
from matplotlib.image import imread

path_photo = '정원용.png'
my_photo = imread(path_photo)

plt.imshow(my_photo)
plt.show()
