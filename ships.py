import json, sys
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD
from PIL import Image
from matplotlib import pyplot as plt

# Download data from JSON object
f = open('C:/Users/Claire/ships-in-satellite-imagery/shipsnet.json')
dataset = json.load(f)
f.close()

input_data = np.array(dataset['data']).astype('uint8')
output_data = np.array(dataset['labels']).astype('uint8')
def describe(input_data,output_data):print('Total number of images: {}'.format(len(input_data))),print('Number of NoShip Images: {}'.format(np.sum(output_data==0))),print('Number of Ship Images: {}'.format(np.sum(output_data==1))),print('Percentage of positive images: {:.2f}%'.format(100*np.mean(output_data))),print('Image shape (Width, Height, Channels): {}'.format(input_data[0].shape))
describe(input_data,output_data)

input_data.shape

n_bands = 3 # data colour bands (RGB) in image
weight = 80
height = 80
X = input_data.reshape([-1, n_bands, weight, height])
X[0].shape

# Plot data
# Data are multispectral images which consist of the 3 visual primary colour bands (red, green, blue) 
# The 3 bands combined produce a "true colour" image, resembling closely what is observed by human eyes 
# We will plot each image in each indivudal colour band
pic = X[1]

red_band = pic[0]
green_band = pic[1]
blue_band = pic[2]

plt.figure(2, figsize = (5*3, 5*1))
plt.set_cmap('jet')

# show each channel
plt.subplot(1, 3, 1)
plt.imshow(red_band)

plt.subplot(1, 3, 2)
plt.imshow(green_band)

plt.subplot(1, 3, 3)
plt.imshow(blue_band)
    
plt.show()

output_data.shape
output_data

np.bincount(output_data)

# Preparing data
# output encoding
y = np_utils.to_categorical(output_data, 2)
# shuffle all indexes
indexes = np.arange(2800)
np.random.shuffle(indexes)
X_train = X[indexes].transpose([0,2,3,1])
y_train = y[indexes]
# normalization
X_train = X_train / 255

# Training network
np.random.seed(42)
# network design
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', input_shape=(80, 80, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) #40x40
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) #20x20
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) #10x10
model.add(Dropout(0.25))

model.add(Conv2D(32, (10, 10), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) #5x5
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax'))

# Optimization setup
sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
    optimizer=sgd,
    metrics=['accuracy'])

# Training
model.fit(X_train, 
    y_train,
    batch_size=32,
    epochs=18,
    validation_split=0.2,
    shuffle=True,
    verbose=2)

# Download image
image = Image.open('C:/Users/Claire/sfbay/sfbay_1.png')
pix = image.load()

n_bands = 3
width = image.size[0]
height = image.size[1]

# Create vector
picture_vector = []
for band in range(n_bands):
    for y in range(height):
        for x in range(width):
            picture_vector.append(pix[x, y][band])
            
picture_vector = np.array(picture_vector).astype('uint8')
picture = picture_vector.reshape([n_bands, height, width]).transpose(1, 2, 0)

#Show the image we are searching for ships on
plt.figure(1, figsize = (15, 30))
plt.subplot(3, 1, 1)
plt.imshow(picture)
plt.show()

picture = picture.transpose(2,0,1)

# Search on the image
def cutting(x, y):
    area_study = np.arange(3*80*80).reshape(3, 80, 80)
    for i in range(80):
        for j in range(80):
            area_study[0][i][j] = picture[0][y+i][x+j]
            area_study[1][i][j] = picture[1][y+i][x+j]
            area_study[2][i][j] = picture[2][y+i][x+j]
    area_study = area_study.reshape([-1, 3, 80, 80])
    area_study = area_study.transpose([0,2,3,1])
    area_study = area_study / 255
    sys.stdout.write('\rX:{0} Y:{1}  '.format(x, y))
    return area_study

def not_near(x, y, s, coordinates):
    result = True
    for e in coordinates:
        if x+s > e[0][0] and x-s < e[0][0] and y+s > e[0][1] and y-s < e[0][1]:
            result = False
    return result

def show_ship(x, y, acc, thickness=5):   
    for i in range(80):
        for ch in range(3):
            for th in range(thickness):
                picture[ch][y+i][x-th] = -1

    for i in range(80):
        for ch in range(3):
            for th in range(thickness):
                picture[ch][y+i][x+th+80] = -1
        
    for i in range(80):
        for ch in range(3):
            for th in range(thickness):
                picture[ch][y-th][x+i] = -1
        
    for i in range(80):
        for ch in range(3):
            for th in range(thickness):
                picture[ch][y+th+80][x+i] = -1
                
step = 10; coordinates = []
for y in range(int((height-(80-step))/step)):
    for x in range(int((width-(80-step))/step) ):
        area = cutting(x*step, y*step)
        result = model.predict(area)
        if result[0][1] > 0.90 and not_near(x*step,y*step, 88, coordinates):
            coordinates.append([[x*step, y*step], result])
            print(result)
            plt.imshow(area[0])
            plt.show()
            
for e in coordinates:
    show_ship(e[0][0], e[0][1], e[1][0][1])
    
picture = picture.transpose(1,2,0)
picture.shape

#Show the 'ships' identified on the image
plt.figure(1, figsize = (15, 30))

plt.subplot(3,1,1)
plt.imshow(picture)

plt.show()


