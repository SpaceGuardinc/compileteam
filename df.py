import pandas as pd

# Указываем путь к файлу Excel
file_path = 'C:/Users/miham/PycharmProjects/compileteam/dataset.xlsx'


# Загружаем данные из файла Excel в DataFrame
df = pd.read_excel(file_path, sheet_name='Лист1')  # Укажите имя листа, если нужно

# Выводим первые несколько строк для проверки
print(df.head())
