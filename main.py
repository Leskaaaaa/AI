import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Dense

x = np.array([
    [1, 2],
    [2, 1],
    [3, 4],
    [4, 3],
    [5, 0],
    [0, 5],
    [6, 1],
    [1, 6],
], dtype=float)

y = x[:, 0] + 0.5 * x[:, 1] + 2

model = keras.Sequential()
model.add(Dense(units=1, input_shape=(2,), activation='linear'))

model.compile(optimizer=keras.optimizers.Adam(0.1), loss='mean_squared_error')

log = model.fit(x, y, epochs=500, verbose=False)

plt.plot(log.history['loss'])
plt.title('График функции потерь')
plt.xlabel('Эпохи')
plt.ylabel('Ошибка')
plt.grid(True)
plt.show()

test = np.array([[7, 3]], dtype=float)
print("Предсказание для [7, 3]:", model.predict(test))

print("Веса и смещение:")
print(model.get_weights())
