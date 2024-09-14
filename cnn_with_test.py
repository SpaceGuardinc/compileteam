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
features = features.reshape(features.shape[1], features.shape[0], 1)

# Определение модели с изменением размеров ядра и шага
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(32, 1, activation='relu', input_shape=(features.shape[1], 1)),
    tf.keras.layers.MaxPooling1D(1),
    tf.keras.layers.Conv1D(64, 1, activation='relu'),
    tf.keras.layers.MaxPooling1D(1),
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

# Визуализация процесса обучения (графики точности и потерь)
plt.figure(figsize=(12, 6))

# График точности
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Точность')
plt.plot(history.history['val_accuracy'], label='Валидационная точность')
plt.xlabel('Эпоха')
plt.ylabel('Точность')
plt.title('Точность на обучении и валидации')
plt.legend()

# График потерь
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Потери')
plt.plot(history.history['val_loss'], label='Валидационные потери')
plt.xlabel('Эпоха')
plt.ylabel('Потери')
plt.title('Потери на обучении и валидации')
plt.legend()

plt.tight_layout()
plt.show()

# Сохранение модели в файл формата HDF5
model.save('trained_model.h5')
print("Модель сохранена в файл 'trained_model.h5'")

# Загрузка обученной модели
loaded_model = tf.keras.models.load_model('trained_model.h5')

# Подготовка данных для предсказания
# Пример данных для двух человек
new_data = pd.DataFrame({
    'Конкретные тесты': ['Креативность', 'Механическая понятливость'],
    'Результат конкретного': [75, 70],
    'Геймификация': ['Креативность', 'Механическая понятливость'],
    'Результат геймификации': [82, 92],
    'Абстрактный тип личности': ['Реалистический', 'Реалистический'],
    'Результат суммы': [73.5, 76.5],
    'Итоговый результат по абстрактному типу личности': [223.5, 223.5]
})

# Преобразование данных в числовой формат (например, нормализация, преобразование категориальных данных)
new_features = []
for column in new_data.columns[1:]:
    try:
        new_features.append(new_data[column].astype(float).values)
    except ValueError:
        continue

new_features = np.array(new_features)
new_features = new_features / 255.0
new_features = new_features.reshape(new_features.shape[1], new_features.shape[0], 1)

# Предсказание
predictions = loaded_model.predict(new_features)

# Интерпретация предсказаний
predicted_labels = label_encoder.inverse_transform(np.argmax(predictions, axis=1))
print("Предсказанные метки для новых данных:", predicted_labels)

# Интерпретация: Подходят ли два человека для командной работы
# Это зависит от логики вашей задачи. Например, вы можете проверить, одинаковы ли типы личности у обоих людей.
if predicted_labels[0] == predicted_labels[1]:
    print("Эти два человека подходят для командной работы.")
else:
    print("Эти два человека могут не подходить для командной работы.")
