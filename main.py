import pandas as pd
import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Загрузка данных из Excel файла
file_path = 'C:/Users/miham/PycharmProjects/compileteam/dataset.xlsx'
data = pd.read_excel(file_path)

# Подготовка данных
data = data.dropna(subset=['Итоговый результат по абстрактному типу личности'])
features = data[['Результат конкретного', 'Результат геймификации']]
labels = data['Итоговый результат по абстрактному типу личности']

# Преобразование данных в numpy массивы
X = features.to_numpy()
y = labels.to_numpy()

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Нормализация данных
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Создание модели нейронной сети
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

# Компиляция модели
model.compile(optimizer='adam', loss='mean_squared_error')

# Обучение модели
history = model.fit(X_train, y_train, epochs=100, batch_size=10, validation_split=0.2)

# Оценка модели
loss = model.evaluate(X_test, y_test)
print(f'Loss: {loss}')

# Пример предсказания
predictions = model.predict(X_test)
print(predictions[:5])


# Функция для ручной настройки весов и пересчета результатов
def manual_adjustment(test_result, game_result, test_weight, game_weight):
    adjusted_result = test_result * test_weight + game_result * game_weight
    return adjusted_result


# Пример использования функции настройки весов
test_weight = float(input("Введите вес для теста: "))
game_weight = float(input("Введите вес для геймификации: "))
adjusted_results = manual_adjustment(X_test[:, 0], X_test[:, 1], test_weight, game_weight)
print(adjusted_results[:5])

# График ошибки на обучающей и валидационной выборках
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# График предсказанных и реальных значений
plt.figure(figsize=(12, 6))
plt.scatter(y_test, predictions, color='blue', label='Predicted vs Actual')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', label='Ideal Fit')
plt.title('Predicted vs Actual Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.show()