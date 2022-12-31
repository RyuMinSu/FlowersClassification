import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import cv2

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import warnings; warnings.filterwarnings("ignore")


basePath = r"E:\python\flowers" #파일 가져오기
foldList = os.listdir(basePath)

flowerDict = dict(filepath=[], filename=[], label=[])
for folder in foldList:
    # print("\nfolder:", folder)
    foldPath = os.path.join(basePath, folder)
    fileName = os.listdir(foldPath)
    # print(f"file numbers:{len(fileName)} in {folder} folder")
    for file in fileName:
        filePath = os.path.join(foldPath, file)
        flowerDict["filepath"].append(filePath)
        flowerDict["filename"].append(file)
        flowerDict["label"].append(folder)
df = pd.DataFrame(flowerDict)
# print("df.shape:", df.shape)
# print(df.head())


imgsize = 150 #array로 변환하여 X, y로 분리
X = []
y = df["label"].values.tolist()
for fpath in tqdm(df["filepath"]):
    imgArray = cv2.imread(fpath, cv2.IMREAD_COLOR)
    imgArray = cv2.resize(imgArray, (imgsize, imgsize))
    X.append(imgArray)
# print(f"X length: {len(X)}")
# print(f"y length: {len(y)}")


#random 이미지 그리기
# fig = plt.figure(figsize=(10, 10))
# for i in range(6):
#     ranNum = np.random.randint(0, len(y))
#     plt.subplot(3, 2, i+1)
#     plt.imshow(X[ranNum])
#     plt.title(y[ranNum], size=15)
# plt.show()
# plt.tight_layout()


y = LabelEncoder().fit_transform(y) #labelencoder
y = to_categorical(y, 5) #onehot
X = np.array(X)
X = X/255.0 #scaling
print(f"X shape: {X.shape}")

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42) #split data
# print(f"x_train shape: {x_train.shape}")
# print(f"x_test shape: {x_test.shape}")
# print(f"y_train shape: {y_train.shape}")
# print(f"y_test shape: {y_test.shape}")



model = Sequential() #model 생성

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu', input_shape = (150,150,3)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
 

model.add(Conv2D(filters =96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters = 96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(5, activation = "softmax"))

model.compile(optimizer=Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])

redLr = ReduceLROnPlateau(monitor="val_acc", patience=3, verbose=1, factor=0.1) #과적합 예방 조기종료
dataGen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=10,
    zoom_range = 0.1,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=False) #과적합 예방 데이터증강
dataGen.fit(x_train)

batch_size=128
epochs=50
history = model.fit_generator(
    dataGen.flow(x_train, y_train, batch_size=batch_size),
    epochs = epochs, validation_data=(x_test, y_test),
    verbose=1, steps_per_epoch=x_train.shape[0] //batch_size)#학습

plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="test")
plt.title("Model accuracy")
plt.ylabel("accuracy", fontsize=10)
plt.xlabel("epochs", fontsize=10)
plt.legend()
plt.show()


CLASSES = ["daisy", "dandelion", "rose", "sunflower", "tulip"]
pred = model.predict(x_test) #예측
pred_num = np.argmax(pred, axis=1) #예측 한 레이블
invy_test = np.argmax(y_test, axis=1) #y_test 레이블

confMat = confusion_matrix(invy_test, pred_num) #confusionMatrix


plt.figure(figsize=(8, 6))
sns.heatmap(confMat, cmap="Greens", annot=True, fmt="d", xticklabels=CLASSES, yticklabels=CLASSES, cbar=True, cbar_kws={"shrink": .5})
plt.title("Flower detection Confusion Matrix", size=20)
plt.xlabel("prediction")
plt.ylabel("actual")
plt.show()