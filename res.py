from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import merge
from keras.layers import AveragePooling2D, Activation, Convolution2D, Dense, Dropout, Flatten, Input
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.optimizers import SGD
from scipy.io import loadmat
from sklearn.model_selection import train_test_split



def conv_bn(inputs, n_filter, n_row, r_col, strides=(1, 1)):
        c = Convolution2D(n_filter, n_row, r_col, subsample=strides, border_mode="same", init="he_normal")(inputs)
        b = BatchNormalization(mode=0, axis=2)(c)
        
        return b

def conv_bn_relu(inputs, n_filter, n_row, r_col, strides=(1, 1)):
        r = conv_bn(inputs, n_filter, n_row, r_col, strides)
        
        return Activation("relu")(r)

def resnet_basic(inputs, n_filter):
        c1 = conv_bn_relu(inputs, n_filter, 3, 3)
        c2 = conv_bn(c1, n_filter, 3, 3)
        p = merge([c2, inputs], mode="sum")
        
        return Activation("relu")(p)

def resnet_basic_inc(inputs, n_filter, strides=(2, 2)):
        c1 = conv_bn_relu(inputs, n_filter, 3, 3, strides)
        c2 = conv_bn(c1, n_filter, 3, 3)
        s = conv_bn(inputs, n_filter, 1, 1, strides)
        p = merge([c2, s], mode="sum")
        
        return Activation("relu")(p)

def resnet_basic_stack(inputs, n_filter, n_layer):
        layer = inputs
        for _ in range(n_layer):
                layer = resnet_basic(layer, n_filter)
        return layer


filters = (16, 32, 64)
n_stack_layer = 3 # ResNet20
input_layer = Input(shape=(3, 32, 32))
block_1 = conv_bn_relu(input_layer, filters[0], 3, 3)
stack_1 = resnet_basic_stack(block_1, filters[0], n_stack_layer)

block_2 = resnet_basic_inc(stack_1, filters[1])
stack_2 = resnet_basic_stack(block_2, filters[1], n_stack_layer-1)

block_3 = resnet_basic_inc(stack_2, filters[2])
stack_3 = resnet_basic_stack(block_3, filters[2], n_stack_layer-1)

# Global average pooling and output
pool = AveragePooling2D(pool_size=(8, 8))(stack_3)
flat = Flatten()(pool)
output_layer = Dense(10, activation="softmax")(flat)
net = Model(input=input_layer, output=output_layer)

if __name__ == "__main__":
        DATA_DIR = "data/cifar-10-batches-mat/data_batch_1.mat"
        batch = loadmat(DATA_DIR)

        data, labels = batch["data"].astype(float).reshape(len(batch["data"]), 3, 32, 32), np_utils.to_categorical(batch["labels"])
        x_train, x_test, y_train, y_test = train_test_split(data, labels)
        epochs = 25
        #lrate = 0.01
        #decay = 0
        #sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
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
        net.fit_generator(datagen.flow(x_train, y_train, batch_size=32, shuffle=True),
                nb_epoch=epochs,
                validation_data=(x_test, y_test),
                samples_per_epoch=x_train.shape[0])

