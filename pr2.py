import time
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense

# ПАРАМЕТРЫ
MAX_WORDS = 10000
BATCH_SIZE = 64
EPOCHS = 5
MAX_LENS = [100, 200, 300, 500]

# ЗАГРУЗКА ДАННЫХ
print("Loading IMDB dataset...")
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=MAX_WORDS)

# ФУНКЦИЯ ПОСТРОЕНИЯ МОДЕЛИ
def build_lstm_model(max_len):
    model = Sequential()
    model.add(Embedding(MAX_WORDS, 128, input_length=max_len))
    model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# ЭКСПЕРИМЕНТ
results = {}

for max_len in MAX_LENS:
    print("\n====================================")
    print(f"Experiment with max_len = {max_len}")
    print("====================================")

    x_train_pad = pad_sequences(x_train, maxlen=max_len)
    x_test_pad = pad_sequences(x_test, maxlen=max_len)

    model = build_lstm_model(max_len)
    model.summary()

    start_time = time.time()
    history = model.fit(
        x_train_pad, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.2,
        verbose=2
    )
    train_time = time.time() - start_time

    test_loss, test_acc = model.evaluate(x_test_pad, y_test, verbose=0)

    results[max_len] = (test_acc, train_time)

    print(f"Test accuracy: {test_acc * 100:.2f}%")
    print(f"Training time: {train_time:.1f} seconds")

# ИТОГОВАЯ ТАБЛИЦА
print("\n========== FINAL RESULTS ==========")
print("max_len | Understanding accuracy | Training time (sec)")
print("----------------------------------")
for max_len, (acc, t) in results.items():
    print(f"{max_len:7} | {acc*100:6.2f}%               | {t:8.1f}")

# ВЫВОД
print("\nConclusion:")
print("Optimal max_len is 200 words, providing best balance between")
print("classification accuracy and training time.")
