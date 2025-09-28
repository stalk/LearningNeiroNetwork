import torch
import torch.nn as nn

# Шаг 1: Определение архитектуры нейросети
# Наша сеть будет очень простой: один входной слой, один скрытый и один выходной.
# Она будет принимать изображение 28x28 пикселей и выдавать 10 значений (для цифр от 0 до 9).
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),  # Входной слой: 784 пикселя
            nn.ReLU(),              # Функция активации
            nn.Linear(512, 512),    # Скрытый слой
            nn.ReLU(),
            nn.Linear(512, 10)      # Выходной слой: 10 классов (цифры от 0 до 9)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# Шаг 2: Создание экземпляра модели и перемещение на GPU
# Здесь мы говорим PyTorch использовать твою видеокарту (RTX 4090).
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Используем {device} устройство.")

model = SimpleNN().to(device)
print(model)

# Шаг 3: Проверка модели
# Создадим "пустое" изображение, чтобы посмотреть, как модель его обработает.
# Размер изображения - 28x28 пикселей.
X = torch.rand(1, 28, 28, device=device)
logits = model(X)
print(f"Размер выходного тензора: {logits.size()}")

logits = torch.tensor([[0.1, 0.05, 0.8, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0]])

# Находим индекс максимального значения
predicted_class_index = torch.argmax(logits)

print(f"Индекс самого высокого 'голоса': {predicted_class_index.item()}")
print(f"Предсказанная цифра: {predicted_class_index.item()}")