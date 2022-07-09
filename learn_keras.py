
import tensorflow
import keras
from keras.layers import Convolution2D, Flatten, Dense 
from keras.layers import Dropout, MaxPooling2D, Input
from keras.models import Sequential

from keras.utils.vis_utils import plot_model



#first method of creating a sequential model
model = Sequential()
model.add(Convolution2D(512, kernel_size=(3,3), activation='relu', padding="same", input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.5))

#second method of creating a sequential model
model1 = Sequential([
    Convolution2D(512, kernel_size=(3,3), activation='relu', padding="same", input_shape=(28, 28, 1)),
    MaxPooling2D((2,2), strides=(2,2)),
    Flatten(),
    Dense(10, activation='relu'),
    Dropout(0.5)
])

keras.utils.plot_model(model, to_file='model.png', show_layer_names=True)


# functional API model

from keras.models import Model


input1 = Input(shape = (28,28,1))
Conv1 = Convolution2D(512, kernel_size=(3,3), activation='relu', padding="same")(input1)
Maxpool1 = MaxPooling2D((2,2), strides=(2,2))(Conv1)
Flatten1 = Flatten()(Maxpool1)
Dense1 = Dense(10, activation='relu')(Flatten1)
Dropout1 = Dropout(0.5)(Dense1)
model2 = Model(input1, Dropout1)

keras.utils.plot_model(model2, to_file='model2.png', show_layer_names=True)