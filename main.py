import keras
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib
# inline

from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.datasets import mnist

# Загрузка данных из датасета mnist и разделяем датасет
# на тренировочную выборку из массива картинок
# 28 на 28 разрешением. объем массива 60к экземпляров.
(X_train, y_train), (X_test, y_test) = mnist.load_data()

plt.imshow(X_train[12], cmap='binary')
plt.axis('off')

# нормируем пиксели, чтобы они были от 0 до 1

X_train = X_train / 255
X_test = X_test / 255

# векторизируем массивы чисел, чтобы получить один единственный
# массив из 0 и 1, который будет представлять каждое из цифр.
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# иициализируем нашу модель
model = Sequential()

# создаем первый слой, в нём будет 32 нейрона, также указываем
# форму входимых данных.
model.add(Dense(32, activation='relu', input_shape=X_train[0].shape))

# создаем еще 5 слоёв
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
# model.add(Dense(512, activation='relu'))
# model.add(Dense(1024, activation='relu'))

# добавляем данные в вектор
model.add(Flatten())

# сравниваем вектор с вектором
model.add(Dense(10, activation='sigmoid'))

# компилируем модель
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# тренируем модель
model.fit(X_train, y_train, epochs=2)

k = 1
plt.imshow(X_test[k], cmap='binary')
plt.axis('off')
plt.show()
print(y_test[k])
print(
    model.predict(np.array([X_test[k]]))
)

k = 2
plt.imshow(X_test[k], cmap='binary')
plt.axis('off')
plt.show()
print(y_test[k])
print(
    model.predict(np.array([X_test[k]]))
)

k = 6
plt.imshow(X_test[k], cmap='binary')
plt.axis('off')
plt.show()
print(y_test[k])
print(
    model.predict(np.array([X_test[k]]))
)
