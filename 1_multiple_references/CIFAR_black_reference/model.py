# https://www.cs.toronto.edu/~kriz/cifar.html
import os

from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import load_model

# # Ensure we don't use too much GPU memory
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto(allow_soft_placement=True,gpu_options = tf.GPUOptions(allow_growth=True))
# set_session(tf.Session(config=config))


def cifar_cnn():
    # Model parameters
    batch_size = 32
    num_classes = 10
    epochs = 100
    num_predictions = 20
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    if not os.path.isdir(save_dir): os.makedirs(save_dir)
    model_name = 'keras_cifar10_trained_model.h5'
    model_path = os.path.join(save_dir, model_name)

    # The data, split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # Preprocess data
    x_train = x_train.astype('float32')/255
    x_test  = x_test.astype('float32')/255

    if not os.path.exists(model_path):

        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same',
                         input_shape=x_train.shape[1:]))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))

        # initiate RMSprop optimizer
        opt = keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)

        # Let's train the model using RMSprop
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                  validation_data=(x_test, y_test), shuffle=True)

        # Save model and weights
        model.save(model_path)
        print('Saved trained model at %s ' % model_path)

        scores = model.evaluate(x_test, y_test, verbose=1)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])
    else:

        model = load_model(model_path)
        # Score trained model.
        scores = model.evaluate(x_test, y_test, verbose=1)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])
    return(model)