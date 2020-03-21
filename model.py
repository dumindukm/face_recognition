

# read images and create train data , test data  e.g: image name , age , gender, ethnicity

# create HDFC file using train data


# model creation

# prediction
# timeout /t 3700 /NOBREAK > NUL && shutdown /h

# [age] is an integer from 0 to 116, indicating the age
# [gender] is either 0 (male) or 1 (female)
# [race] is an integer from 0 to 4, denoting White, Black, Asian, Indian, and Others (like Hispanic, Latino, Middle Eastern).


# https://github.com/jangedoo/age-gender-race-prediction

# python model.py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import glob
import os

from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
import cv2

input_dim = 200 * 200

image_list = []
image_data = []
labels = []
num_pixels = 200*200  # 64*64*3

for file in glob.glob("UTKFace/*.jpg"):
    # print(file)
    image_list.append(file)
    # image_data.append(np.reshape( cv2.imread(file), num_pixels))

    resized = cv2.resize(cv2.imread(file, 0), (200, 200),
                         interpolation=cv2.INTER_AREA)
    image_data.append(resized.reshape(num_pixels,)/255)
    file_name = os.path.basename(file)

    # print(os.path.basename(file).split('_'))
    labels.append(file_name.split('_')[0:1])

# print('image_list',image_list)
# print('labels',labels)

scaler = MinMaxScaler()
scaler.fit(labels)
labels_scaled = scaler.transform(labels)

X_train, X_test, y_train, y_test = train_test_split(
    image_data, labels_scaled, test_size=0.33, random_state=42)


print(np.shape(X_train))
print(np.shape(y_train))

# print(X_train)


X_train = np.reshape(X_train, (np.shape(X_train)[0], num_pixels))
X_test = np.reshape(X_test, (np.shape(X_test)[0], num_pixels))

y_train = np.reshape(y_train, (np.shape(y_train)[0], 1))
y_test = np.reshape(y_test, (np.shape(y_test)[0], 1))


print('X_test', np.shape(X_test))
print(np.shape(y_train))


model = Sequential()
# model.add(Flatten(input_shape=(64, 64,3)))
model.add(Dense(512, input_shape=(num_pixels,)))
# model.add(Dense(32,input_shape=(200*200*3,)))
model.add(Dense(512, activation='relu'))
# model.add(Dense(512, activation='relu'))
model.add(Dropout(.3))
model.add(Dense(256, activation='relu'))
model.add(Dropout(.1))
model.add(Dense(1, activation='linear'))


model.compile(metrics=['mean_absolute_error'], optimizer=Adam(
    lr=0.0001), loss='mean_squared_error')  # mean_squared_error


# model = load_model('Checkpoints/best_model_20200321.h5')
modelCheckpoint = ModelCheckpoint('Checkpoints/best_model_20200321.h5',
                                  monitor='val_loss', verbose=0, save_best_only=True,  mode='auto', period=1)
es = EarlyStopping(monitor='val_loss', mode='min', patience=50, min_delta=.002)

model.fit([X_train], y_train, epochs=2000, validation_split=0.3,
          batch_size=200, callbacks=[modelCheckpoint, es])

# model.save('best_model.h5')# 2000 +2000 +2000+2000+2000+2000+2000+3000+1000


# print('X_test', np.shape(X_test[]))
# cv2.namedWindow('pre')        # Create a named window
# cv2.moveWindow('pre', 240, 230)
# cv2.imshow('pre', X_test[0])
# cv2.waitKey()
# result = model.predict([[X_test[0]]])

# print('evaluate', model.evaluate(
#     scaler.inverse_transform[X_test], scaler.inverse_transform[y_test]))

for index, imae in enumerate(X_test):
    result = model.predict([[imae]])
    print("{} {}".format(scaler.inverse_transform(
        [[result[0][0]]]), scaler.inverse_transform([y_test[index]])))

result = model.predict([[X_test[0]]])
print('result', scaler.inverse_transform(result))
print('actual Y', scaler.inverse_transform([y_test[0]]))


y_t_scaled = scaler.inverse_transform(y_test)
print('eveluate', model.evaluate(
    X_test, y_t_scaled))

print('done')
