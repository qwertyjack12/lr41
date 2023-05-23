import matplotlib.pyplot as plt
import tensorflow as tf
from keras.datasets import mnist
import numpy as np
import cv2
seed = 256    # Для воспроизведения результатов
tf.random.set_seed(seed)
np.random.seed(seed)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_t=[]
image_size=(32, 32)    # Для полно связной сети используйте (20,20)
for img in X_train:
  X_t.append(cv2.resize(img,image_size))

x_test=[]
for img in X_test:
  x_test.append(cv2.resize(img, image_size))

X_t=np.array(X_t)
x_test=np.array(x_test)
X_t = X_t.astype("float32") / 255   # Нормализация изображений
x_test = x_test.astype("float32") / 255
x_train = np.expand_dims(X_t, -1)   # увеличение размерности
x_test = np.expand_dims(x_test, -1)
y_train = tf.keras.utils.to_categorical(y_train, 10)    # one hot encoding
y_test = tf.keras.utils.to_categorical(y_test, 10)

batch_size = 32
epochs = 3
opt = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.0,
                            nesterov=False, name='SGD')
NumerNN = tf.keras.models.Sequential()
NumerNN.add(tf.keras.layers.Conv2D(6, 5, activation='tanh',
                                  input_shape=x_train.shape[1:]))
NumerNN.add(tf.keras.layers.AveragePooling2D(2))
NumerNN.add(tf.keras.layers.Conv2D(16, 5, activation='tanh'))
NumerNN.add(tf.keras.layers.AveragePooling2D(2))
NumerNN.add(tf.keras.layers.Conv2D(120, 5, activation='tanh'))
NumerNN.add(tf.keras.layers.Flatten())
NumerNN.add(tf.keras.layers.Dense(84, activation='tanh'))
NumerNN.add(tf.keras.layers.Dense(10, activation='softmax'))
NumerNN.summary()
NumerNN.compile(loss='categorical_crossentropy', optimizer=opt,
               metrics=["accuracy"])

NumerNN.fit(x_train, y_train, batch_size, epochs, validation_split=0.2)
NumerNN.evaluate(x_test, y_test)

x = np.expand_dims(x_test[0], axis=0)
res = NumerNN.predict(x)
print(res)
print(np.argmax(res))

plt.imshow(x_test[0], cmap=plt.cm.binary)
plt.show()