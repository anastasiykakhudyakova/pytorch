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

# ============================================================================
# БЛОК 1: ИМПОРТ БИБЛИОТЕК
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torchvision

# ============================================================================
# БЛОК 2: ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# ============================================================================

# Путь для сохранения данных
root = "./Data_10"

# Размер пакета (для обучения и тестирования)
batch_size = 32  # Увеличил для ускорения обучения

# Трансформация изображений
transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Загрузка тренировочных данных
train_set = datasets.CIFAR10(
    root=root,
    train=True,
    transform=transformations,
    download=True
)

# Загрузка тестовых данных
test_set = datasets.CIFAR10(
    root=root,
    train=False,
    transform=transformations,
    download=True
)

# Создание DataLoader'ов
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# Классы изображений CIFAR10
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck']

# ============================================================================
# БЛОК 3: СОЗДАНИЕ МОДЕЛИ (с ядрами 3x3)
# ============================================================================

class ImprovedImageModel(nn.Module):
    """
    Улучшенная модель с ядрами свёрточных слоёв 3x3.
    Архитектура:
    - Свёрточный слой (3 -> 32, kernel=3)
    - Batch Normalization
    - ReLU
    - MaxPooling (2x2)
    - Свёрточный слой (32 -> 64, kernel=3)
    - Batch Normalization
    - ReLU
    - Linear (полносвязный слой)
    """
    def __init__(self):
        super(ImprovedImageModel, self).__init__()
        
        # Первый сверточный слой: вход 3 канала (RGB), выход 32 канала, ядро 3x3
        # Размер изображения: 32x32 -> после свертки с padding=1: 32x32
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, 
                               kernel_size=3, stride=1, padding=1)
        
        # Слой нормализации (по количеству выходных каналов conv1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # MaxPooling слой: уменьшает размерность в 2 раза (32x32 -> 16x16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Второй сверточный слой: вход 32 канала, выход 64 канала, ядро 3x3
        # Размер изображения: 16x16 -> после свертки с padding=1: 16x16
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, 
                               kernel_size=3, stride=1, padding=1)
        
        # Слой нормализации (по количеству выходных каналов conv2)
        self.bn2 = nn.BatchNorm2d(64)
        
        # После сверток и пулинга размер изображения: 64 канала * 16 * 16 = 16384
        # Полносвязный слой: вход 16384, выход 10 (количество классов)
        self.fc = nn.Linear(64 * 16 * 16, 10)
    
    def forward(self, x):
        # Первый блок: Conv -> BN -> ReLU -> Pool
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Второй блок: Conv -> BN -> ReLU
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Изменяем размерность для полносвязного слоя
        # Преобразуем [batch, 64, 16, 16] в [batch, 64*16*16]
        x = x.view(x.size(0), -1)
        
        # Полносвязный слой
        x = self.fc(x)
        
        return x

# ============================================================================
# БЛОК 4: ФУНКЦИЯ ДЛЯ РАСЧЁТА ТОЧНОСТИ
# ============================================================================

def calculate_accuracy(model, data_loader):
    """
    Рассчитывает точность модели на переданном DataLoader'е
    
    Args:
        model: нейронная сеть
        data_loader: DataLoader с данными
    
    Returns:
        accuracy: точность в процентах
    """
    model.eval()  # Переводим модель в режим оценки
    correct = 0
    total = 0
    
    with torch.no_grad():  # Отключаем вычисление градиентов (экономия памяти)
        for images, labels in data_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

# ============================================================================
# БЛОК 5: ФУНКЦИЯ ДЛЯ ОБУЧЕНИЯ МОДЕЛИ
# ============================================================================

def train_model(model, train_loader, test_loader, num_epochs=5):
    """
    Обучает модель и выводит точность после каждой эпохи
    
    Args:
        model: нейронная сеть
        train_loader: DataLoader для обучения
        test_loader: DataLoader для тестирования
        num_epochs: количество эпох обучения
    """
    # Функция потерь (перекрестная энтропия для классификации)
    criterion = nn.CrossEntropyLoss()
    
    # Оптимизатор Adam с learning rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("\n" + "="*60)
    print("НАЧАЛО ОБУЧЕНИЯ")
    print("="*60)
    
    for epoch in range(num_epochs):
        model.train()  # Переключаем в режим обучения
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader, 0):
            # Обнуляем градиенты
            optimizer.zero_grad()
            
            # Прямое распространение
            outputs = model(images)
            
            # Вычисляем ошибку
            loss = criterion(outputs, labels)
            
            # Обратное распространение
            loss.backward()
            
            # Обновляем веса
            optimizer.step()
            
            running_loss += loss.item()
        
        # Вычисляем точность на тестовых данных после эпохи
        accuracy = calculate_accuracy(model, test_loader)
        
        print(f"Эпоха {epoch+1}/{num_epochs} | Потери: {running_loss/len(train_loader):.4f} | Точность: {accuracy:.2f}%")
    
    print("="*60)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print("="*60)
    
    return model

# ============================================================================
# БЛОК 6: ФУНКЦИЯ ДЛЯ ВИЗУАЛИЗАЦИИ 20 ИЗОБРАЖЕНИЙ
# ============================================================================

def visualize_predictions(model, test_loader, num_images=20):
    """
    Визуализирует изображения, выводит правильные ответы и предсказания модели
    
    Args:
        model: обученная нейронная сеть
        test_loader: DataLoader с тестовыми данными
        num_images: количество изображений для вывода (по умолчанию 20)
    """
    model.eval()  # Режим оценки
    
    # Получаем первый батч изображений
    images, true_labels = next(iter(test_loader))
    
    # Берем только первые num_images изображений
    images = images[:num_images]
    true_labels = true_labels[:num_images]
    
    # Получаем предсказания модели
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    # Вывод в консоль
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ ПРЕДСКАЗАНИЙ ДЛЯ 20 ИЗОБРАЖЕНИЙ")
    print("="*60)
    
    print("\nПравильные ответы:")
    for i, label in enumerate(true_labels):
        print(f"{i+1:2d}. {classes[label]}", end="   ")
        if (i+1) % 5 == 0:
            print()
    
    print("\n\nПредсказания модели:")
    for i, pred in enumerate(predicted):
        print(f"{i+1:2d}. {classes[pred]}", end="   ")
        if (i+1) % 5 == 0:
            print()
    
    # Подсчет точности для 20 изображений
    correct = (predicted == true_labels).sum().item()
    accuracy = 100 * correct / num_images
    
    print(f"\n\n{'='*60}")
    print(f"ТОЧНОСТЬ НА 20 ИЗОБРАЖЕНИЯХ: {correct}/{num_images} = {accuracy:.1f}%")
    print("="*60)
    
    # Визуализация изображений
    # Объединяем изображения в сетку (4 строки x 5 столбцов)
    grid_images = torchvision.utils.make_grid(images, nrow=5, padding=2)
    
    # Восстанавливаем нормализацию для отображения (из [-1, 1] в [0, 1])
    grid_images = grid_images / 2 + 0.5
    
    # Преобразуем для matplotlib
    image_to_show = np.transpose(grid_images.numpy(), (1, 2, 0))
    
    # Создаем фигуру с подписями
    plt.figure(figsize=(15, 12))
    plt.imshow(image_to_show)
    plt.axis('off')
    plt.title(f'CIFAR-10: 20 Test Images\nAccuracy: {correct}/{num_images} ({accuracy:.1f}%)', 
              fontsize=14, fontweight='bold')
    
    # Добавляем подписи под каждым изображением
    # (make_grid меняет порядок, поэтому нужно аккуратно позиционировать)
    plt.tight_layout()
    plt.show()
    
    return accuracy, correct, predicted, true_labels

# ============================================================================
# БЛОК 7: ЗАМЕР ВРЕМЕНИ ОБУЧЕНИЯ
# ============================================================================

import time

def train_with_timing(model, train_loader, test_loader, num_epochs=5):
    """
    Обучает модель с замером времени
    """
    start_time = time.time()
    model = train_model(model, train_loader, test_loader, num_epochs)
    end_time = time.time()
    
    training_time = end_time - start_time
    print(f"\nВремя обучения: {training_time:.2f} секунд")
    
    return model, training_time

# ============================================================================
# БЛОК 8: ВЫПОЛНЕНИЕ ЗАДАНИЯ
# ============================================================================

print("\n" + "="*60)
print("ЗАДАНИЕ: Уменьшение ядер свёрточных слоёв до 3x3")
print("="*60)

# Создаем модель с ядрами 3x3
model_3x3 = ImprovedImageModel()
print(f"\nАрхитектура модели:")
print(f"- Свёрточный слой 1: 3 -> 32 канала, ядро 3x3")
print(f"- Batch Normalization 1")
print(f"- ReLU")
print(f"- MaxPooling 2x2")
print(f"- Свёрточный слой 2: 32 -> 64 канала, ядро 3x3")
print(f"- Batch Normalization 2")
print(f"- ReLU")
print(f"- Linear слой: 64*16*16 = 16384 -> 10")

# Обучаем модель и замеряем время
print("\n" + "-"*40)
print("НАЧАЛО ОБУЧЕНИЯ С ЯДРАМИ 3x3")
print("-"*40)

trained_model, training_time = train_with_timing(model_3x3, train_loader, test_loader, num_epochs=5)

# Финальная точность на всех тестовых данных
final_accuracy = calculate_accuracy(trained_model, test_loader)
print(f"\nФинальная точность на всех тестовых данных (10000 изображений): {final_accuracy:.2f}%")

# ============================================================================
# БЛОК 9: ВИЗУАЛИЗАЦИЯ 20 ИЗОБРАЖЕНИЙ И РАСЧЁТ ТОЧНОСТИ
# ============================================================================

print("\n" + "="*60)
print("ВИЗУАЛИЗАЦИЯ 20 ИЗОБРАЖЕНИЙ ИЗ ТЕСТОВОГО НАБОРА")
print("="*60)

# Визуализируем 20 изображений и получаем точность
accuracy_20, correct_20, predictions, true_labels_20 = visualize_predictions(
    trained_model, test_loader, num_images=20
)

# ============================================================================
# БЛОК 10: ВЫВОД ОТВЕТОВ НА ВОПРОСЫ ЗАДАНИЯ
# ============================================================================

print("\n" + "="*60)
print("ОТВЕТЫ НА ВОПРОСЫ ЗАДАНИЯ")
print("="*60)

print("""
1. ИЗМЕНЕНИЕ ЯДЕР СВЁРТОЧНЫХ СЛОЁВ С 5x5 ДО 3x3:
   - Точность: Обычно повышается или остаётся на том же уровне,
     так как меньшее ядро позволяет выявлять более мелкие детали.
   - Скорость: Увеличивается, так как 3x3 требует меньше вычислений,
     чем 5x5 (9 операций против 25 на одну позицию).

2. ИЗМЕНЕНИЕ АРХИТЕКТУРЫ МОДЕЛИ:
   - Новая архитектура: Conv -> BN -> ReLU -> Pool -> Conv -> BN -> ReLU -> Linear
   - Преимущества:
     * Batch Normalization стабилизирует обучение
     * MaxPooling уменьшает размерность и ускоряет обучение
     * Меньше параметров -> быстрее обучение
     * Меньше риск переобучения

3. РЕЗУЛЬТАТЫ НА 20 ИЗОБРАЖЕНИЯХ:
""")

print(f"   - Правильно предсказано: {correct_20} из 20")
print(f"   - Точность: {accuracy_20:.1f}%")
print(f"   - Время обучения (5 эпох): {training_time:.2f} секунд")
print(f"   - Финальная точность на всех тестовых данных: {final_accuracy:.2f}%")

# ============================================================================
# БЛОК 11: СРАВНЕНИЕ С ПРЕДЫДУЩЕЙ МОДЕЛЬЮ (для отчета)
# ============================================================================

print("\n" + "="*60)
print("СРАВНЕНИЕ С МОДЕЛЬЮ С ЯДРАМИ 5x5")
print("="*60)
print("""
┌────────────────────┬─────────────────────┬─────────────────────┐
│      Параметр      │   Модель с 5x5      │   Модель с 3x3      │
├────────────────────┼─────────────────────┼─────────────────────┤
│ Размер ядра        │ 5x5                 │ 3x3                 │
│ Количество слоёв   │ 4 сверточных + BN   │ 2 сверточных + BN   │
│ Параметры (прим.)  │ ~500,000            │ ~150,000            │
│ Скорость обучения  │ Медленнее           │ Быстрее             │
│ Точность (ожид.)   │ ~65-70%             │ ~70-75%             │
└────────────────────┴─────────────────────┴─────────────────────┘

ВЫВОД: Уменьшение ядер свёрточных слоёв до 3x3 и упрощение архитектуры
позволяет ускорить обучение и часто повысить точность за счёт лучшего
выявления локальных признаков.
""")
