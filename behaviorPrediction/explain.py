import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# --- Часть 1: Подготовка данных ---
# Эта функция загружает, очищает и преобразует табличные данные
def prepare_data(file_path):
    """
    Загружает данные из CSV, выполняет их очистку и преобразование для ML.

    Аргументы:
        file_path (str): Путь к CSV-файлу с данными.

    Возвращает:
        X_scaled (np.ndarray): Массив признаков (features), отмасштабированных.
        y (np.ndarray): Массив целевой переменной (target), выживших.
    """
    # 1. Загрузка данных с помощью pandas
    df = pd.read_csv(file_path)

    # 2. Обработка пропущенных значений.
    # Заполняем пропуски в колонке 'Age' медианным значением,
    # так как нейросеть не умеет работать с отсутствующими данными.
    df['Age'] = df['Age'].fillna(df['Age'].median())

    # 3. Преобразование категориальных данных в числовые.
    # Используем pd.get_dummies для One-Hot Encoding.
    # Например, 'Sex' ('male', 'female') превратится в две колонки 'Sex_female' и 'Sex_male' (мы оставим только одну, т.к. они взаимоисключающие)
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

    # 4. Удаление ненужных для обучения колонок.
    # 'PassengerId', 'Name' и 'Ticket' не несут полезной информации для модели.
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

    # 5. Разделение данных на признаки (X) и целевую переменную (y).
    # X - это все колонки, кроме 'Survived'. y - это 'Survived'.
    X = df.drop('Survived', axis=1)
    y = df['Survived']

    # 6. Масштабирование признаков.
    # Приводим все числовые признаки к одному масштабу (среднее = 0, стандартное отклонение = 1).
    # Это помогает нейросети обучаться быстрее.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y.values


# --- Часть 2: Архитектура нейронной сети ---
# Класс, который определяет структуру нашей нейросети.
# Наследование от nn.Module - обязательное условие для PyTorch.
class TabularNN(nn.Module):
    def __init__(self, num_features):
        super(TabularNN, self).__init__()
        # nn.Sequential позволяет объединить слои в единую последовательность.
        self.stack = nn.Sequential(
            # nn.Linear - линейный слой. Первый аргумент - размер входа, второй - размер выхода.
            nn.Linear(num_features, 64),
            # nn.ReLU - функция активации, добавляет нелинейность.
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            # Выходной слой. Для бинарной классификации нам нужен 1 выходной нейрон.
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # Метод forward определяет, как данные "протекают" через сеть.
        # Здесь мы просто передаем данные через нашу последовательность слоев.
        return self.stack(x)


# --- Часть 3: Функции для обучения и тестирования ---
# Цикл обучения модели
def train_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()  # Переводим модель в режим обучения
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Squeeze() убирает лишние размерности (например, [32, 1] становится [32]),
        # чтобы они соответствовали друг другу для функции потерь.
        y = y.squeeze()

        pred = model(X).squeeze()
        loss = loss_fn(pred, y)

        # Выполняем обратное распространение ошибки и обновляем веса
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 50 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# Цикл тестирования модели
def test_loop(dataloader, model, loss_fn, device):
    model.eval()  # Переводим модель в режим оценки (без обучения)
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():  # Отключаем вычисление градиентов для экономии памяти и ускорения
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y = y.squeeze()
            pred = model(X).squeeze()
            test_loss += loss_fn(pred, y).item()

            # Логика для бинарной классификации.
            # Если предсказание (без Sigmoid) > 0, то считаем, что это класс 1, иначе - класс 0.
            predicted = (pred > 0.0).float()
            correct += (predicted == y).sum().item()

    test_loss /= num_batches
    accuracy = correct / size
    print(f"Test Error: \n Accuracy: {(100 * accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# --- Часть 4: Запуск скрипта ---
if __name__ == "__main__":
    # Загружаем и готовим данные
    X_data, y_data = prepare_data('train.csv')

    # Разделяем на тренировочный и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.2, random_state=42
    )

    # Преобразуем данные в тензоры PyTorch
    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).float().unsqueeze(1)
    X_test_tensor = torch.from_numpy(X_test).float()
    y_test_tensor = torch.from_numpy(y_test).float().unsqueeze(1)

    # Создаем Datasets и DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Определяем устройство для обучения (GPU или CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Используем {device} устройство.")

    # Определяем модель. Количество признаков берется из данных.
    num_features = X_train_tensor.shape[1]
    model = TabularNN(num_features).to(device)

    # Определяем функцию потерь для бинарной классификации.
    # BCEWithLogitsLoss - это оптимизированная версия для бинарных задач.
    loss_fn = nn.BCEWithLogitsLoss()

    # Определяем оптимизатор Adam, который эффективно управляет скоростью обучения.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Запускаем основной цикл обучения
    epochs = 20
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, device)
        test_loop(test_dataloader, model, loss_fn, device)

    print("Готово!")