import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Загрузка данных из файла XLSX
data = pd.read_excel('dataset.xlsx', sheet_name='Лист1')

# Преобразование строковых меток в целочисленные идентификаторы
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(data.iloc[:, 0].values)  # метки классов

# Преобразование числовых признаков в numpy массив
features = []
for column in data.columns[1:]:
    try:
        features.append(data[column].astype(float).values)
    except ValueError:
        continue

features = np.array(features)

# Нормализация данных
features = features / 255.0

# Приведение к необходимой размерности для сверточной сети (добавление канала)
features = features.reshape(features.shape[0], features.shape[1], 1)

# Определение модели
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(32, 3, activation='relu', input_shape=(features.shape[1], 1)),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Conv1D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
])

# Компиляция модели
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Обучение модели
history = model.fit(features, labels, epochs=20, batch_size=32, validation_split=0.2)

# Визуализация процесса обучения (графики accuracy и loss)
plt.figure(figsize=(12, 6))

# График accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

# График loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Сохранение модели в файл формата HDF5
model.save('trained_model.h5')
print("Модель сохранена в файл 'trained_model.h5'")

