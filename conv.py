from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, Dense, Flatten, MaxPooling2D, Dropout
from keras.utils import np_utils
from keras.optimizers import SGD
from scipy.io import loadmat
from sklearn.model_selection import train_test_split



net = Sequential()
net.add(Convolution2D(32, 3, 3, activation="relu", input_shape=(3, 32, 32), border_mode="same"))
net.add(Convolution2D(32, 3, 3, activation="relu", border_mode="same"))
net.add(MaxPooling2D(pool_size=(2, 2)))
net.add(Dropout(0.25))


net.add(Convolution2D(64, 3, 3, activation="relu", border_mode="same"))
net.add(Convolution2D(64, 3, 3, activation="relu", border_mode="same"))
net.add(MaxPooling2D(pool_size=(2, 2)))
net.add(Dropout(0.25))

net.add(Flatten())
net.add(Dense(512, activation="relu"))
net.add(Dropout(0.5))
net.add(Dense(10, activation="softmax"))


if __name__ == "__main__":
        DATA_DIR = "data/cifar-10-batches-mat/data_batch_1.mat"
        batch = loadmat(DATA_DIR)

        data, labels = batch["data"].astype(float).reshape(len(batch["data"]), 3, 32, 32), np_utils.to_categorical(batch["labels"])
        x_train, x_test, y_train, y_test = train_test_split(data, labels)
        net.compile(loss="categorical_crossentropy",
                optimizer="adam",
                metrics=["accuracy"])
                
                
        datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False)
                
        datagen.fit(x_train, augment=True) 
        net.fit_generator(datagen.flow(x_train, y_train, batch_size=128, shuffle=True),
                nb_epoch=20,
                validation_data=(x_test, y_test),
                samples_per_epoch=x_train.shape[0])
