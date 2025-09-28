import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch

def prepare_data(file_path):
    # Загружаем данные
    df = pd.read_csv(file_path)

    # Заполняем пропуски в числовых колонках медианой
    df['Age'] = df['Age'].fillna(df['Age'].median())

    # Преобразуем категориальные данные в числовые (One-Hot Encoding)
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

    # Удаляем ненужные колонки
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

    # Разделяем данные на признаки (X) и целевую переменную (y)
    X = df.drop('Survived', axis=1)
    y = df['Survived']

    # Масштабируем числовые признаки
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Преобразуем данные в тензоры PyTorch
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)

    # Возвращаем подготовленные данные
    return X_tensor, y_tensor

# Пример использования функции
# Загрузка и подготовка данных
X, y = prepare_data('train.csv')

# Теперь у нас есть тензоры X и y
print(f"Размер тензора признаков: {X.shape}")
print(f"Размер тензора целевой переменной: {y.shape}")

# Мы получили 8 признаков, что соответствует нашему TabularNN