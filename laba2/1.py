# Импорт необходимых библиотек
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Определяем путь для сохранения данных
root = "./Data_10"

# 2. Размер пакета (batch size)
batch_size = 10

# 3. Описываем трансформацию изображений:
#    - Преобразование в тензор (переводит пиксели [0..255] в [0..1])
#    - Нормализация: (x - mean) / std, чтобы привести значения к диапазону [-1; +1]
transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # (mean, std)
])

# 4. Загрузка тренировочных данных (train=True)
train_set = datasets.CIFAR10(
    root=root,
    train=True,           # тренировочный набор (50 000 изображений)
    transform=transformations,
    download=True         # скачать, если нет в root
)

# 5. DataLoader для тренировочных данных (с перемешиванием)
train_data_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True          # случайный порядок для лучшего обучения
)

# 6. Загрузка тестовых данных (train=False)
test_set = datasets.CIFAR10(
    root=root,
    train=False,          # тестовый набор (10 000 изображений)
    transform=transformations,
    download=True
)

# 7. DataLoader для тестовых данных (без перемешивания для стабильной оценки)
test_data_loader = DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False         # фиксированный порядок для тестирования
)

# Проверка: посмотрим размеры загруженных данных
print(f"Размер тренировочного набора: {len(train_set)} изображений")
print(f"Размер тестового набора: {len(test_set)} изображений")
print(f"Batch size: {batch_size}")
print(f"Количество батчей в тренировке: {len(train_data_loader)}")
print(f"Количество батчей в тесте: {len(test_data_loader)}")

# Пример одного батча
data_iter = iter(train_data_loader)
images, labels = next(data_iter)
print(f"\nФорма одного батча изображений: {images.shape}")  # [batch, channels, height, width]
print(f"Диапазон значений после нормализации: [{images.min():.2f}, {images.max():.2f}]")
