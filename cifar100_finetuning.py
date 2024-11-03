import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, UpSampling2D, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam

def create_model():
    model = Sequential()
    model.add(UpSampling2D((2,2)))
    model.add(UpSampling2D((2,2)))
    model.add(UpSampling2D((2,2)))
    model.add(base_model)
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(100, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['acc'])
    return model
    

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
print(base_model.summary())

from tensorflow.keras.datasets import cifar100
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

y_train = to_categorical(y_train)

print(x_train.shape)
print(x_test.shape)

# Finetuning model on CIFAR10 training data
model = create_model()
model.fit(x_train, y_train, epochs=10, batch_size=32)

y_pred_test = model.predict(x_test)
from sklearn.metrics import classification_report
print(classification_report(np.argmax(y_pred_test,1), y_test))
