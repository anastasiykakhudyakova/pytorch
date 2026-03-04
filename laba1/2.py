'''Задания для самостоятельной работы
Создайте нейросеть со следующей архитектурой: 5 нейронов на входе, три скрытых слоя, 1 нейрон на выходе (бинарная классификация).
При этом каждый скрытый слой должен пропускаться через различных функции активации.
Подумайте, какая должна быть функция активация на выходе в предыдущем задании, с учётом того, что это бинарная классификация. Добавьте её.
Попробуйте поменять оптимизатор. Какой лучше работает в Вашем случае: Adam или SGD?'''

# Импортируем необходимые библиотеки
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
#  Создание модели для бинарной классификации
class ThreeHiddenLayerBinaryModel(nn.Module):
    """
    Класс нейросети с тремя скрытыми слоями для бинарной классификации
    """
    def __init__(self, input_size, hidden1_size, hidden2_size, hidden3_size, output_size):
        """
        Конструктор модели
        input_size: количество входных нейронов (5)
        hidden1_size: количество нейронов в 1-м скрытом слое
        hidden2_size: количество нейронов во 2-м скрытом слое
        hidden3_size: количество нейронов в 3-м скрытом слое
        output_size: количество выходных нейронов (1 для бинарной классификации)
        """
        # Вызываем конструктор родительского класса nn.Module
        super(ThreeHiddenLayerBinaryModel, self).__init__()
        
        # ПЕРВЫЙ СКРЫТЫЙ СЛОЙ с функцией активации ReLU
        # Линейный слой: преобразует входные данные (5 нейронов) в hidden1_size нейронов
        self.fc1 = nn.Linear(input_size, hidden1_size)
        # Функция активации ReLU для первого скрытого слоя
        # ReLU(x) = max(0, x) - помогает бороться с проблемой затухания градиента
        self.relu1 = nn.ReLU()
        
        # ВТОРОЙ СКРЫТЫЙ СЛОЙ с функцией активации Tanh
        # Линейный слой: от hidden1_size до hidden2_size нейронов
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        # Функция активации Tanh (гиперболический тангенс)
        # Преобразует значения в диапазон [-1, 1], хорошо подходит для центрирования данных
        self.tanh = nn.Tanh()
        
        # ТРЕТИЙ СКРЫТЫЙ СЛОЙ с функцией активации LeakyReLU
        # Линейный слой: от hidden2_size до hidden3_size нейронов
        self.fc3 = nn.Linear(hidden2_size, hidden3_size)
        # LeakyReLU - улучшенная версия ReLU, допускает небольшие отрицательные значения
        # LeakyReLU(x) = max(0.01x, x) - помогает избежать "умирающих нейронов"
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        
        # ВЫХОДНОЙ СЛОЙ для бинарной классификации
        # Линейный слой: от hidden3_size до output_size (1 нейрон)
        self.fc4 = nn.Linear(hidden3_size, output_size)
        
        # ВНИМАНИЕ: Sigmoid не добавляем здесь, так как будем использовать 
        # BCELoss (Binary Cross Entropy Loss) или BCEWithLogitsLoss
        
    def forward(self, x):
        """
        Прямой проход по сети
        x: входные данные (batch_size, input_size)
        """
        # Пропускаем через первый скрытый слой с ReLU
        out = self.fc1(x)
        out = self.relu1(out)
        
        # Пропускаем через второй скрытый слой с Tanh
        out = self.fc2(out)
        out = self.tanh(out)
        
        # Пропускаем через третий скрытый слой с LeakyReLU
        out = self.fc3(out)
        out = self.leaky_relu(out)
        
        # Выходной слой (без активации, если используем BCEWithLogitsLoss)
        out = self.fc4(out)
        
        return out

# ================================================
# ЗАДАНИЕ 2: Правильная функция активации для выхода
# ================================================

"""
ОТВЕТ: Для бинарной классификации на выходе должна быть функция активации SIGMOID

Почему Sigmoid?
1. Преобразует выходное значение в вероятность от 0 до 1
2. Формула: σ(x) = 1 / (1 + e^(-x))
3. Позволяет интерпретировать результат как вероятность принадлежности к классу 1
4. Для бинарной классификации порог обычно 0.5: если >0.5 - класс 1, иначе класс 0

Два варианта реализации:

ВАРИАНТ А - Явно добавляем Sigmoid:
self.sigmoid = nn.Sigmoid()
...
out = self.sigmoid(self.fc4(out))

ВАРИАНТ Б - Используем BCEWithLogitsLoss (рекомендуется):
Этот вариант объединяет Sigmoid и вычисление ошибки, что численно более стабильно
"""

# ================================================
# ЗАДАНИЕ 3: Сравнение оптимизаторов (Adam vs SGD)
# ================================================

def create_and_train_model(optimizer_type='adam'):
    """
    Функция для создания и обучения модели с разными оптимизаторами
    optimizer_type: 'adam' или 'sgd'
    """
    
    # Параметры модели (согласно заданию: 5 входов, 1 выход)
    input_size = 5
    hidden1_size = 10  # можно экспериментировать с размерами
    hidden2_size = 8
    hidden3_size = 6
    output_size = 1  # один выход для бинарной классификации
    
    # Параметры обучения
    learning_rate = 0.01  # для SGD нужен может быть другой learning rate
    num_epochs = 200
    
    # Создаем модель
    model = ThreeHiddenLayerBinaryModel(
        input_size, hidden1_size, hidden2_size, hidden3_size, output_size
    )
    
    # Функция потерь для бинарной классификации
    # BCEWithLogitsLoss объединяет Sigmoid + Binary Cross Entropy
    # Это более стабильно численно, чем явный Sigmoid + BCELoss
    criterion = nn.BCEWithLogitsLoss()
    
    # Выбор оптимизатора
    if optimizer_type == 'adam':
        # Adam - адаптивный оптимизатор с индивидуальным learning rate для параметров
        # Хорошо работает по умолчанию, быстро сходится
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        print(f"\n--- Используем оптимизатор ADAM (lr={learning_rate}) ---")
    else:
        # SGD - стохастический градиентный спуск
        # Может требовать тщательного подбора learning rate, но иногда дает лучшее качество
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        print(f"\n--- Используем оптимизатор SGD с momentum (lr={learning_rate}) ---")
    
    # Генерируем синтетические данные для бинарной классификации
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Обучающие данные: 200 образцов
    n_samples = 200
    
    # Входные данные: случайные числа
    x_train = np.random.rand(n_samples, input_size).astype(np.float32)
    
    # Генерируем метки для бинарной классификации (0 или 1)
    # Создаем некоторую зависимость от входных данных для более реалистичной задачи
    # Например, сумма первых двух признаков > 1 дает класс 1
    y_train = (x_train[:, 0] + x_train[:, 1] > 1).astype(np.float32)
    # Добавляем немного шума
    noise_mask = np.random.random(n_samples) < 0.1
    y_train[noise_mask] = 1 - y_train[noise_mask]
    
    # Преобразуем в тензоры PyTorch
    x_train = torch.from_numpy(x_train)
    # Для BCEWithLogitsLoss нужна форма [batch_size, 1]
    y_train = torch.from_numpy(y_train).view(-1, 1)
    
    # Тестовые данные
    x_test = np.random.rand(50, input_size).astype(np.float32)
    y_test = (x_test[:, 0] + x_test[:, 1] > 1).astype(np.float32)
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test).view(-1, 1)
    
    # Списки для записи истории обучения
    train_losses = []
    
    # Обучение модели
    print("Эпоха\tПотери")
    print("-" * 20)
    
    for epoch in range(num_epochs):
        # Обнуляем градиенты
        optimizer.zero_grad()
        
        # Прямой проход: получаем логиты (невероятности!)
        outputs = model(x_train)
        
        # Вычисляем ошибку
        loss = criterion(outputs, y_train)
        
        # Обратное распространение
        loss.backward()
        
        # Обновляем веса
        optimizer.step()
        
        # Сохраняем значение потерь
        train_losses.append(loss.item())
        
        # Выводим каждые 20 эпох
        if (epoch + 1) % 20 == 0:
            print(f"{epoch+1:4d}\t{loss.item():.6f}")
    
    # Оценка на тестовых данных
    with torch.no_grad():  # отключаем вычисление градиентов
        test_outputs = model(x_test)
        # Применяем Sigmoid для получения вероятностей
        probabilities = torch.sigmoid(test_outputs)
        # Предсказания (0 или 1) с порогом 0.5
        predictions = (probabilities > 0.5).float()
        
        # Считаем точность
        accuracy = (predictions == y_test).float().mean().item()
        
    print(f"\nТочность на тестовых данных: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Показываем несколько примеров предсказаний
    print("\nПримеры предсказаний (вероятности):")
    for i in range(min(5, len(x_test))):
        prob = probabilities[i].item()
        pred_class = predictions[i].item()
        true_class = y_test[i].item()
        print(f"Образец {i+1}: вероятность класса 1 = {prob:.4f} -> класс {int(pred_class)} (истинный: {int(true_class)})")
    
    return model, train_losses

# ================================================
# ЗАПУСК И СРАВНЕНИЕ ОПТИМИЗАТОРОВ
# ================================================

print("=" * 50)
print("СРАВНЕНИЕ ОПТИМИЗАТОРОВ ДЛЯ БИНАРНОЙ КЛАССИФИКАЦИИ")
print("=" * 50)

# Обучаем с Adam
model_adam, losses_adam = create_and_train_model('adam')

print("\n" + "=" * 50)

# Обучаем с SGD
model_sgd, losses_sgd = create_and_train_model('sgd')

# ================================================
# АНАЛИЗ РЕЗУЛЬТАТОВ
# ================================================

print("\n" + "=" * 50)
print("ВЫВОДЫ ПО СРАВНЕНИЮ ОПТИМИЗАТОРОВ:")
print("=" * 50)
print("""
Обычно Adam работает лучше SGD по нескольким причинам:

1. Адаптивный learning rate: Adam подстраивает скорость обучения для каждого параметра
2. Быстрая сходимость: обычно требует меньше эпох для достижения хорошего результата
3. Устойчивость к выбору гиперпараметров: Adam часто хорошо работает с настройками по умолчанию

Однако SGD с моментом может давать лучшее качество на некоторых задачах, 
особенно когда нужно достичь очень высокой точности, но требует больше эпох
и тщательного подбора learning rate.

В данном эксперименте вы должны увидеть, что Adam быстрее снижает ошибку
и достигает хорошей точности на тестовых данных.
""")

# ================================================
# ДОПОЛНИТЕЛЬНЫЕ ЭКСПЕРИМЕНТЫ ДЛЯ САМОСТОЯТЕЛЬНОЙ РАБОТЫ
# ================================================

"""
ЗАДАНИЯ ДЛЯ САМОСТОЯТЕЛЬНОГО ИССЛЕДОВАНИЯ:

1. Попробуйте разные размеры скрытых слоев (например, [20, 15, 10], [8, 8, 8])
2. Измените функции активации (попробуйте ELU, SELU, PReLU)
3. Добавьте Dropout слои для регуляризации
4. Попробуйте разные learning rates (0.001, 0.01, 0.1)
5. Добавьте L2 регуляризацию (weight_decay в оптимизаторе)
6. Попробуйте разные функции потерь (BCELoss с явным Sigmoid)
"""

# Пример модифицированной модели с Dropout для экспериментов
class AdvancedBinaryModel(nn.Module):
    """Расширенная модель для самостоятельных экспериментов"""
    
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.2):
        super(AdvancedBinaryModel, self).__init__()
        
        self.layers = nn.ModuleList()
        
        # Создаем скрытые слои динамически
        prev_size = input_size
        for i, hidden_size in enumerate(hidden_sizes):
            # Линейный слой
            self.layers.append(nn.Linear(prev_size, hidden_size))
            
            # Разные функции активации для разных слоев
            if i % 3 == 0:
                self.layers.append(nn.ReLU())
            elif i % 3 == 1:
                self.layers.append(nn.Tanh())
            else:
                self.layers.append(nn.LeakyReLU(0.1))
            
            # Dropout для регуляризации
            self.layers.append(nn.Dropout(dropout_rate))
            
            prev_size = hidden_size
        
        # Выходной слой
        self.output_layer = nn.Linear(prev_size, output_size)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        return x

