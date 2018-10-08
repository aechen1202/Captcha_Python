# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 16:40:57 2018

@author: a639351
"""
from PIL import Image
import glob
import os
def getImageData(path):
    x_image = []
    y_label =[]
    for filename in glob.glob(path):
        img=Image.open(filename).convert('L')
        x_image.append(np.array(img))
        y_label.append(os.path.split(filename)[1][0])
        img.close()
    return (np.asarray(x_image),y_label)


#資料預處理
from keras.utils import np_utils
import numpy as np
(x_train_image,y_train_label)=getImageData('C:\CaptchaImg\Train/*.jpg')
x_Train=x_train_image.reshape(x_train_image.shape[0], 2500).astype('float32')
x_Train_normalize = x_Train / 255
y_Train_OneHot = np_utils.to_categorical(y_train_label)

(x_Test_image,y_Test_label)=getImageData('C:\CaptchaImg\Test/*.jpg')
x_Test=x_Test_image.reshape(x_Test_image.shape[0], 2500).astype('float32')
x_Test_normalize = x_Test / 255
y_Test_OneHot = np_utils.to_categorical(y_Test_label)


#建立模型
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
model=Sequential()
#input
model.add(Dense(units=5000,input_dim=2500,kernel_initializer='normal',activation='relu'))
#Dropout
model.add(Dropout(0.5))
#將「隱藏層2」加入模型
model.add(Dense(units=5000,kernel_initializer='normal',activation='relu'))
model.add(Dropout(0.5))
#output
model.add(Dense(units=10,kernel_initializer='normal',activation='softmax'))
print(model.summary())

#訓練模型
from keras import optimizers
sgd = optimizers.SGD(lr=0.001, decay=0.0005, momentum=0.09, nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
train_history=model.fit(x=x_Train_normalize,y=y_Train_OneHot,validation_split=0.3,epochs=10, batch_size=200,verbose=2)

#以圖形顯示訓練過程
import matplotlib.pyplot as plt
def show_train_history(train_history,train,validation):
     plt.plot(train_history.history[train])
     plt.plot(train_history.history[validation])
     plt.title('Train History')
     plt.ylabel(train)
     plt.xlabel('Epoch')
     plt.legend(['train', 'validation'], loc='upper left')
     plt.show()
show_train_history(train_history,'acc','val_acc')
show_train_history(train_history,'loss','val_loss')

#評估模型準確率
scores = model.evaluate(x_Test_normalize,y_Test_OneHot)
print('accuracy=',scores[1])

##進行預測
prediction=model.predict_classes(x_Test)
print(prediction)
def plot_images_labels_prediction(images,labels,predection,idx,num=10):
    fig = plt.gcf()
    fig.set_size_inches(12,14)
    if num>25: num=25
    for i in range(0,num):
        ax=plt.subplot(5,5,1+i)
        ax.imshow(images[idx],cmap='binary')
        title="label=" +str(labels[idx])
        if len(predection)>0:
            title="label=" +str(predection[idx])
        ax.set_title(title,fontsize=10)
        ax.set_xticks([]);ax.set_yticks([])
        idx+=1
    plt.show()
plot_images_labels_prediction(x_Test_image,y_Test_label,prediction,0,x_Test_image.shape[0])

