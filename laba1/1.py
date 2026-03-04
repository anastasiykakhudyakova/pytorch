'''Установите PyTorch. Проверьте правильность установки'''
import torch  # Импортируем библиотеку PyTorch
print(f"Версия PyTorch: {torch.__version__}")# Проверяем версию PyTorch для подтверждения правильности установки

''' Создайте 2 тензора 3x3 со случайными значениями. Выведите их.'''
tensor1 = torch.rand(3, 3)  # rand создает тензор со случайными значениями от 0 до 1
tensor2 = torch.rand(3, 3)  # Второй тензор для операций
print(tensor1)
print(tensor2)

'''Сложите созданные тензоры и выведите результат.'''
sum_tensor = tensor1 + tensor2  # Поэлементное сложение тензоров
# Альтернатива: torch.add(tensor1, tensor2)
print(sum_tensor)

'''Умножьте первый тензор на второй поэлементно и выведите результат.'''
mult_tensor = tensor1 * tensor2  # Поэлементное умножение (не матричное!)
print(mult_tensor)

'''Транспонируйте второй тензор и выведите результат.'''
transposed_tensor = tensor2.T  # .T транспонирует матрицу (меняет строки и столбцы)
# Альтернатива: torch.transpose(tensor2, 0, 1)
print(transposed_tensor)

'''Найдите среднее значение в каждом тензоре и выведите их.'''
mean1 = tensor1.mean()  # Вычисляем среднее арифметическое всех элементов
mean2 = tensor2.mean()
print(f"Среднее значение первого тензора: {mean1:.4f}")
print(f"Среднее значение второго тензора: {mean2:.4f}")
# или  print(torch.mean(tensor1), torch.mean(tensor2))

'''Найдите максимальное значение в каждом тензоре и выведите их.'''
max1 = tensor1.max()  # Находим максимальный элемент
max2 = tensor2.max()
print(f"Максимальное значение первого тензора: {max1:.4f}")
print(f"Максимальное значение второго тензора: {max2:.4f}")


'''Используя фреймворк PyTorch, создайте нейросеть, которая будет перемножать входные 2 нейрона. Например, если подаёте (3, 4), то выходной нейрон должен быть примерно равен 12. 
В обучающем наборе должно быть 100 примеров. Примечание: набор обучающих данных можно сгенерировать с помощью random и операции перемножения.'''
import torch.nn as nn  # Модуль для создания нейросетей
import numpy as np  # Для работы с массивами
import torch.optim as optim  # Модуль с оптимизаторами
''' Самые популярные оптимизаторы:
optimizer1 = optim.SGD(model.parameters(), lr=0.01)        # Стохастический градиентный спуск
optimizer2 = optim.Adam(model.parameters(), lr=0.001)      # Adam (самый популярный)
optimizer3 = optim.RMSprop(model.parameters(), lr=0.01)    # RMSprop
optimizer4 = optim.Adagrad(model.parameters(), lr=0.01)    # Adagrad
optimizer5 = optim.AdamW(model.parameters(), lr=0.001)     # Adam с Weight Decay'''

# Устанавливаем seed для воспроизводимости результатов
torch.manual_seed(42)
np.random.seed(42)

# Определяем класс нейросети
class MultiplicationNet(nn.Module):
    def __init__(self):
        super(MultiplicationNet, self).__init__()  # Вызываем конструктор родительского класса
        # Создаем слои нейросети:
        # Вход: 2 нейрона (два числа для умножения)
        # Скрытый слой: 10 нейронов для обучения нелинейной зависимости
        self.fc1 = nn.Linear(2, 10)  # Первый полносвязный слой: 2 входа, 10 выходов
        self.relu = nn.ReLU()  # Функция активации ReLU (отрицательные значения превращает в 0)
        self.fc2 = nn.Linear(10, 1)  # Выходной слой: 10 входов, 1 выход (результат умножения)
        
    def forward(self, x):
        # Прямой проход по сети
        x = self.fc1(x)  # Применяем первый слой
        x = self.relu(x)  # Применяем активацию ReLU
        x = self.fc2(x)  # Применяем выходной слой (без активации, так как регрессия)
        return x

# Создаем модель
model = MultiplicationNet()
print(model)

# Функция потерь и оптимизатор
criterion = nn.MSELoss()  # Среднеквадратичная ошибка (для задач регрессии)
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adam оптимизатор с скоростью обучения 0.01

# Генерация обучающих данных
n_samples = 100  # Количество обучающих примеров
# Создаем случайные числа от 1 до 10
X_train = torch.FloatTensor(n_samples, 2).uniform_(1, 10)  # 100 строк, 2 столбца
# Вычисляем правильные ответы (произведение)
y_train = (X_train[:, 0] * X_train[:, 1]).view(-1, 1)  # view(-1,1) превращает в столбец

#Сгенерировано {n_samples} обучающих примеров (первые 5)
for i in range(5):
    print(f"  {X_train[i,0]:.2f} * {X_train[i,1]:.2f} = {y_train[i,0]:.2f}")

# Обучение модели
epochs = 500  # Количество эпох обучения

for epoch in range(epochs):
    # Прямой проход
    outputs = model(X_train)  # Пропускаем входные данные через сеть
    loss = criterion(outputs, y_train)  # Вычисляем ошибку
    
    # Обратный проход и оптимизация
    optimizer.zero_grad()  # Обнуляем градиенты
    loss.backward()  # Вычисляем градиенты
    optimizer.step()  # Обновляем веса
    
    # Выводим прогресс каждые 50 эпох
    if (epoch + 1) % 50 == 0:
        print(f'Эпоха [{epoch+1}/{epochs}], Потери: {loss.item():.6f}')



'''Сгенерируйте 10 проверочных пар чисел. Пропустите их через Вашу нейросеть, выведя все необходимые данные для анализа корректности работы нейросети.)'''
X_test = torch.FloatTensor(10, 2).uniform_(1, 10)  # 10 случайных пар # Генерируем 10 проверочных пар чисел

# Переводим модель в режим оценки (отключает dropout и т.д., если есть)
model.eval()

# Отключаем вычисление градиентов для экономии памяти и ускорения
with torch.no_grad():
    predictions = model(X_test)  # Получаем предсказания модели
    true_values = X_test[:, 0] * X_test[:, 1]  # Вычисляем истинные значения

# Выводим результаты для анализа
print("-" * 60)
print(f"{'a':>6} {'b':>6} {'Предсказание':>12} {'Истина':>8} {'Ошибка':>10} {'Точность%':>10}")
print("-" * 60)

for i in range(10):
    a = X_test[i, 0].item()
    b = X_test[i, 1].item()
    pred = predictions[i, 0].item()
    true = true_values[i].item()
    error = abs(pred - true)
    accuracy = 100 * (1 - error/true) if true != 0 else 0
    
    print(f"{a:6.2f} {b:6.2f} {pred:12.4f} {true:8.2f} {error:10.4f} {accuracy:9.2f}%")

print("-" * 60)



'''Попробуйте изменить количество тренировочных данных. Например, вместо 100 сделать 10, а потом 1000. Как изменился результат?'''
def train_with_samples(n_samples, epochs=300):
    """Функция для обучения модели с разным количеством примеров"""
    
    # Создаем новую модель
    model = MultiplicationNet()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # Генерируем данные
    X = torch.FloatTensor(n_samples, 2).uniform_(1, 10)
    y = (X[:, 0] * X[:, 1]).view(-1, 1)
    
    # Обучаем
    for epoch in range(epochs):
        outputs = model(X)
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Тестируем на фиксированных данных
    test_X = torch.tensor([[3.0, 4.0], [5.0, 6.0], [2.0, 8.0]])
    with torch.no_grad():
        test_pred = model(test_X)
        test_true = test_X[:, 0] * test_X[:, 1]
    
    # Средняя ошибка на тесте
    test_error = torch.mean(torch.abs(test_pred.view(-1) - test_true)).item()
    
    return test_error

# Пробуем с разным количеством данных
print("Сравнение результатов при разном объеме данных:")
print("-" * 50)
print(f"{'Количество данных':<20} {'Средняя ошибка':<20}")
print("-" * 50)

for samples in [10, 100, 1000]:
    error = train_with_samples(samples)
    print(f"{samples:<20} {error:<20.6f}")

print("\nВывод: Чем больше данных, тем меньше ошибка, но обучение дольше.")


'''Сохранение, загрузка'''
# Сохраняем модель
model_path = 'multiplication_model.pth'
torch.save(model.state_dict(), model_path)
# Создаем новую модель (с той же архитектурой)
loaded_model = MultiplicationNet()
# Загружаем сохраненные веса
loaded_model.load_state_dict(torch.load(model_path))
