



# read images and create train data , test data  e.g: image name , age , gender, ethnicity

# create HDFC file using train data


# model creation

# prediction
#timeout /t 3700 /NOBREAK > NUL && shutdown /h

#[age] is an integer from 0 to 116, indicating the age
#[gender] is either 0 (male) or 1 (female)
#[race] is an integer from 0 to 4, denoting White, Black, Asian, Indian, and Others (like Hispanic, Latino, Middle Eastern).


# https://github.com/jangedoo/age-gender-race-prediction

# python model.py
import numpy as np
from sklearn.model_selection import train_test_split
import glob
import os

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Dropout
from keras.optimizers import Adam
import cv2

input_dim = 200 * 200

image_list = []
image_data  = []
labels = []
num_pixels = 200*200 * 3

for file in glob.glob("UTK/*.jpg"):
	#print(file)
	image_list.append(file)
	# image_data.append(np.reshape( cv2.imread(file), num_pixels))
	
	resized = cv2.resize(cv2.imread(file), (64,64), interpolation = cv2.INTER_AREA)
	image_data.append(resized)
	file_name = os.path.basename(file)

	#print(os.path.basename(file).split('_'))
	labels.append( file_name.split('_')[0:1])

print('image_list',image_list)
print('labels',labels)


X_train, X_test, y_train, y_test = train_test_split(image_data,labels, test_size=0.33, random_state=42)

print(np.shape(X_train))
print(np.shape(y_train))

# print(X_train)


X_train = np.reshape(X_train,(np.shape(X_train)[0], 64,64,3)) #X_train.reshape((x_train.shape[0], num_pixels)).astype('float32')
X_test = np.reshape(X_test,(np.shape(X_test)[0],  64,64,3)) #X_test.reshape((X_test.shape[0], num_pixels)).astype('float32')

y_train = np.reshape(y_train, (np.shape(y_train)[0],1))
y_test = np.reshape(y_test, (np.shape(y_test)[0],1))

print('X_test', np.shape(X_test))
print(np.shape(y_train))


model = Sequential()
model.add(Flatten(input_shape=(64, 64,3)))
# model.add(Dense(32,input_shape=(200*200*3,)))
model.add(Dense(1042, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))

model.add(Dense(1))

model.compile(metrics =['accuracy'], optimizer=Adam(lr=0.01), loss='mean_squared_error')#mean_squared_error
model.fit(X_train, y_train, epochs=2500, batch_size=20)

model.save('best_model.h5')

#print('X_test', np.shape(X_test[]))
cv2.namedWindow('pre')        # Create a named window
cv2.moveWindow('pre', 240,230)
cv2.imshow('pre',X_test[0])
cv2.waitKey()
result = model.predict([[X_test[0]]])

# for imae in X_test[1]:
	# cv2.imshow('pre',imae)
	# result = model.predict([imae])


print(result)
print('done')

