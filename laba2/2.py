'''
root="./Data_10"	Локальное хранение	Не скачивать данные при каждом запуске
batch_size=10	Пакетная обработка	Экономия памяти, стабильный градиент
shuffle=True (train)	Перемешивание	Модель не запоминает порядок данных
shuffle=False (test)	Без перемешивания	Воспроизводимость результатов
Нормализация [-1, 1]	Стабилизация	Предотвращение переполнения при обучении
'''

'''
ЗАДАНИЕ:
1. Уменьшите размер ядер свёрточных слоёв, сделав их 3x3. 
   Как изменился результат обучения в плане точности и скорости?
   Примечание: не забудьте поменять другие параметры в модели, иначе обучение не начнётся.

2. Измените модель, сделав её такой: 
   свёрточный слой → слой нормализации → слой ReLU → слой max-pooling → 
   свёрточный слой → слой нормализации → слой ReLU → линейный слой.
Как изменился результат обучения в плане точности и скорости? Примечание: не забудьте поменять другие параметры в модели, иначе обучение не начнётся.
3. Используя тестовый набор изображений, вытащите из него 20 картинок.
   Выведите все 20 изображений на экран, а в консоль выведите правильные ответы и прогноз нейросети.

4. Рассчитайте точность прогноза в процентах на основе 20 изображений из предыдущего задания 
(например, если угадано 15 изображений из 20, то точность равна 75%). Выведите получившуюся точность в консоль.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Загрузка данных CIFAR-10
def load_cifar10_data(batch_size=10, root='./Data_10'):
    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transformations)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_set = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transformations)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# архитектураа
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        # Первый сверточный слой с ядром 3x3
        self.conv1 = nn.Conv2d(3, 12, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        # Второй сверточный слой
        self.conv2 = nn.Conv2d(12, 24, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(24)
        # Слой пулинга
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Линейный слой
        self.fc = nn.Linear(24 * 8 * 8, 10)  # Исправлено: 8x8 вместо 16x16

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(-1, 24 * 8 * 8)  
        x = self.fc(x)
        return x

#
# Функция для оценки точности
def test_accuracy(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# Функция для вывода изображений и прогнозов
def show_images_with_predictions(model, test_loader, classes, num_images=20):
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images[:num_images], labels[:num_images]
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    # Вывод изображений
    images = torchvision.utils.make_grid(images)
    images = images / 2 + 0.5
    np_images = images.numpy()
    plt.imshow(np.transpose(np_images, (1, 2, 0)))
    plt.axis('off')
    plt.show()

    # Вывод правильных ответов и прогнозов
    print("Правильные ответы:", [classes[labels[i]] for i in range(num_images)])
    print("Прогнозы нейросети:", [classes[predicted[i]] for i in range(num_images)])

    # Расчет точности
    correct = (predicted == labels).sum().item()
    accuracy = 100 * correct / num_images
    print(f"Точность на {num_images} изображениях: {accuracy:.2f}%")

# Основная функция
def main():
    # Загрузка данных
    train_loader, test_loader = load_cifar10_data(batch_size=10)

    # Определение модели
    model = CustomCNN()

    # Определение функции потерь и оптимизатора
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # Обучение модели
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Оценка точности после каждой эпохи
        accuracy = test_accuracy(model, test_loader)
        print(f"Эпоха {epoch + 1}, точность: {accuracy:.2f}%")

    # Вывод изображений и прогнозов
    classes = ('самолет', 'автомобиль', 'птица','кот', 'олень', 'собака', 'лягушка', 'лошадь', 'корабль', 'грузовик')
    show_images_with_predictions(model, test_loader, classes, num_images=20)

if __name__ == "__main__":
    main()




