import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# Загрузка данных из файла XLSX для тестирования
test_data = pd.read_excel('test_dataset.xlsx', sheet_name='Sheet1')

# Преобразование строковых меток в целочисленные идентификаторы
label_encoder = LabelEncoder()
test_labels = label_encoder.fit_transform(test_data.iloc[:, 0].values)  # метки классов

# Преобразование числовых признаков в numpy массив
test_features = []
for column in test_data.columns[1:]:
    try:
        test_features.append(test_data[column].astype(float).values)
    except ValueError:
        continue

test_features = np.array(test_features)

# Нормализация данных
test_features = test_features / 255.0

# Приведение к необходимой размерности для сверточной сети (добавление канала)
test_features = test_features.reshape(test_features.shape[0], test_features.shape[1], 1)

# Загрузка обученной модели из файла
model = tf.keras.models.load_model('trained_model.h5')

# Оценка модели на тестовых данных
test_loss, test_accuracy = model.evaluate(test_features, test_labels)

print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
print(f'Test Loss: {test_loss:.4f}')
