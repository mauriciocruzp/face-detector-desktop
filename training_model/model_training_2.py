import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf
from skimage.io import imread_collection
from skimage.transform import resize
from keras.callbacks import ModelCheckpoint

user_folder = "training_model/data/user/*.jpg"
unknown_folder = "training_model/data/unknown/*.jpg"

user_images = imread_collection(user_folder)
unknown_images = imread_collection(unknown_folder)

user_images_n = len(user_images)
unknown_images_n = len(unknown_images)

images = np.append(user_images, unknown_images, axis=0)

plt.imshow(images[0])
print(images[0].shape)


def create_y():
    return [0] * user_images_n + [1] * unknown_images_n


y = create_y()

y = np.array(y)
x = np.array(images)

x = resize(x, (len(images), 64, 64, 3))

plt.imshow(x[0])
print(x[0].shape)

print(x.shape[1:])

model = Sequential()

model.add(Conv2D(200, (3, 3), input_shape=X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(100, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(50, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(50, activation="relu"))
model.add(Dense(2, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

y = to_categorical(y)

y[0]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.30, random_state=0
)

checkpoint = ModelCheckpoint(
    "model.h5", monitor="val_loss", verbose=0, save_best_only=True, mode="auto"
)

history = model.fit(x_train, y_train, epochs=20, callbacks=[checkpoint], validation_split=0.2)

model.save("model.h5")

saved_model_dir = ""
tf.saved_model.save(model, saved_model_dir)

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)
