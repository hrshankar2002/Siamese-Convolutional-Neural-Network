import tensorflow as tf
from tensorflow import keras
from keras import layers

def build_alexnet_model(input_shape=(227, 227, 3), num_classes=17):
    model = keras.Sequential()
    model.add(layers.Conv2D(filters=96, kernel_size=(11, 11),
                            strides=(4, 4), activation="relu",
                            input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
    
    model.add(layers.Conv2D(filters=256, kernel_size=(5, 5),
                            strides=(1, 1), activation="relu",
                            padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
    
    model.add(layers.Conv2D(filters=384, kernel_size=(3, 3),
                            strides=(1, 1), activation="relu",
                            padding="same"))
    model.add(layers.BatchNormalization())
    
    model.add(layers.Conv2D(filters=384, kernel_size=(3, 3),
                            strides=(1, 1), activation="relu",
                            padding="same"))
    model.add(layers.BatchNormalization())
    
    model.add(layers.Conv2D(filters=256, kernel_size=(3, 3),
                            strides=(1, 1), activation="relu",
                            padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation="softmax"))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.optimizers.SGD(learning_rate=0.001),
                  metrics=['accuracy'])
    return model
