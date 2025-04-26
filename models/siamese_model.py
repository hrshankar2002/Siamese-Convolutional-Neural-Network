from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate
from keras.models import Model

def build_siamese_model(input_shape=(224, 224, 3), num_classes=17):
    input_left = Input(shape=input_shape)
    input_right = Input(shape=input_shape)

    conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu')
    pool1 = MaxPooling2D(pool_size=(2, 2))
    conv2 = Conv2D(64, kernel_size=(3, 3), activation='relu')
    pool2 = MaxPooling2D(pool_size=(2, 2))
    flatten = Flatten()

    x1 = conv1(input_left)
    x1 = pool1(x1)
    x1 = conv2(x1)
    x1 = pool2(x1)
    x1 = flatten(x1)

    x2 = conv1(input_right)
    x2 = pool1(x2)
    x2 = conv2(x2)
    x2 = pool2(x2)
    x2 = flatten(x2)

    concatenated = concatenate([x1, x2])
    dense1 = Dense(128, activation='relu')(concatenated)
    output = Dense(num_classes, activation='softmax')(dense1)

    model = Model(inputs=[input_left, input_right], outputs=output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
