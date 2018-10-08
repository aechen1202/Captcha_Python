# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 15:13:13 2018

@author: a639351
"""

from PIL import Image
import glob
import os
def getImageData(path):
    x_image = []
    y_label =[]
    for filename in glob.glob(path):
        img=Image.open(filename)
        #img=Image.open(filename).convert('L')
        x_image.append(np.array(img))
        y_label.append(os.path.split(filename)[1][0])
        img.close()
    return (np.asarray(x_image),y_label)

#資料預處理
from keras.utils import np_utils
import numpy as np
(x_Train,y_train_label)=getImageData('C:\CaptchaImg\Train/*.jpg')
x_Train4D=x_Train.reshape(x_Train.shape[0],50,50,3).astype('float32')
x_Train4D_normalize = x_Train4D /255
y_Train_OneHot = np_utils.to_categorical(y_train_label)

(x_Test,y_Test_label)=getImageData('C:\CaptchaImg\Test/*.jpg')
x_Test4D=x_Test.reshape(x_Test.shape[0],50,50,3).astype('float32')
x_Test4D_normalize = x_Test4D / 255
y_Test_OneHot = np_utils.to_categorical(y_Test_label)

#建立模型
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
model = Sequential()
model.add(Conv2D(filters=16,
                 kernel_size=(5,5),
                 padding='same',
                 input_shape=(50,50,3), 
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(5, 5)))
model.add(Conv2D(filters=36,
                 kernel_size=(5,5),
                 padding='same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))
print(model.summary())

#訓練模型
model.compile(loss='categorical_crossentropy',
              optimizer='adam',metrics=['accuracy']) 
train_history=model.fit(x=x_Train4D_normalize,y=y_Train_OneHot,validation_split=0.2
                        ,epochs=20, batch_size=300,verbose=2)
import matplotlib.pyplot as plt
def show_train_history(train_acc,test_acc):
    plt.plot(train_history.history[train_acc])
    plt.plot(train_history.history[test_acc])
    plt.title('Train History')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
show_train_history('acc','val_acc')
show_train_history('loss','val_loss')

#評估模型準確率
scores = model.evaluate(x_Test4D_normalize , y_Test_OneHot)
print('accuracy=',scores[1])

#預測結果
prediction = model.predict_classes(x_Test4D_normalize)
print(prediction[:x_Test4D_normalize.shape[0]-1])

#查看預測結果
import matplotlib.pyplot as plt
def plot_images_labels_prediction(images,labels,prediction,idx,num):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    #if num>25: num=25 
    for i in range(0, num):
        ax=plt.subplot(200,5, 1+i)
        ax.imshow(images[idx], cmap='binary')

        ax.set_title("label=" +str(labels[idx])+
                     ",predict="+str(prediction[idx])
                     ,fontsize=10) 
        
        ax.set_xticks([]);ax.set_yticks([])        
        idx+=1 
    plt.show()
plot_images_labels_prediction(x_Test,y_Test_label,prediction,0,x_Test.shape[0])