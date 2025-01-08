import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_raws', None)

# задание 1 , чтение
df = pd.read_csv("base.csv", encoding='windows-1251', delimiter=';')
# print(df)

# Задание 2 фильр по помещению и статусу
filtered_df = df[df['ВидПомещения'] == 'жилые помещения'].dropna(subset='СледующийСтатус')

# замена статуса
filtered_df['СледующийСтатус'] = filtered_df['СледующийСтатус'].replace(to_replace=['Свободна', 'Продана'],
                                                                        value=[0, 1])


# print(filtered_df[filtered_df['СледующийСтатус'] == 1 ])
# print(filtered_df)

# Задание 3
def convert_to_number(value):
    # принадлежит ли к str
    if isinstance(value, str):
        value = value.rstrip('к')
        try:
            return float(value)
        except ValueError:
            return np.nan
    return np.nan


filtered_df['ПродаваемаяПлощадь'] = pd.to_numeric(filtered_df['ПродаваемаяПлощадь'], errors='coerce')
filtered_df['СтоимостьНаДатуБрони'] = pd.to_numeric(filtered_df['СтоимостьНаДатуБрони'], errors='coerce')
filtered_df['ФактическаяСтоимостьПомещения'] = pd.to_numeric(filtered_df['ФактическаяСтоимостьПомещения'],
                                                             errors='coerce')

# b) Применяем бинарное кодирование для столбца "ИсточникБрони" (например, ручная -> 1, другой источник -> 0)
# filtered_df['ИсточникБрони'] = filtered_df['ИсточникБрони'].apply(lambda x: 1 if x == 'ручная' else 0)

numeric_columns_mask = filtered_df.apply(lambda x: pd.api.types.is_numeric_dtype(x))
numeric_columns_list = filtered_df.columns[numeric_columns_mask].to_list()
# print(f"Поля, имеющие числовой тип: {numeric_columns_list}")

df_unique = df.apply(lambda x: x.dropna().unique())
for column_name, column_items in df_unique.items():
    if len(column_items) == 2:
        df[column_name] = df[column_name].replace(sorted(column_items), [1, 0])
        print(column_name, " | ", sorted(column_items), " -> ", [1, 0])

# c) Выполняем one-hot кодирование для категориальных признаков, например "Город"
categorical_columns = [
    'Город',
    'Статус лида (из CRM)'
]
filtered_df = pd.get_dummies(filtered_df, columns=categorical_columns, drop_first=True)

print('TYPE')
print(filtered_df['Тип'])
# Функция для обработки значений в столбце 'Тип'
def process_type(value):
    if isinstance(value, str) and value.endswith('к'):  # Проверяем заканчивается ли на 'к'
        try:
            # Удаляем 'к' заменяем ',' на '.' и преобразуем в число
            return float(value[:-1].replace(',', '.'))
        except ValueError:
            return np.nan  # Если не удается преобразовать возвращаем NaN
    return np.nan  # Для всего остального возвращаем NaN

# Применяем функцию к столбцу
filtered_df['Тип'] = filtered_df['Тип'].apply(process_type)

print('TYPE2')
print(filtered_df['Тип'])

# print(filtered_df)
# print(filtered_df.dtypes)

# Задание 4
missing_data = filtered_df.isnull().sum()
print('Количество отсутствующих данных')
print(missing_data)
# замена в скидке на квартиру
filtered_df['СкидкаНаКвартиру'].fillna(0, inplace=True)
# print(filtered_df['СкидкаНаКвартиру'])

# Тип, Продаваемая площадь
filtered_df['Тип'].fillna(filtered_df['Тип'].median(), inplace=True)
filtered_df['ПродаваемаяПлощадь'].fillna(filtered_df['ПродаваемаяПлощадь'].median(), inplace=True)
# print(filtered_df['Тип'])
# print(filtered_df['ПродаваемаяПлощадь'])

# удаление оставшихся и проверка
filtered_df = filtered_df.dropna(subset='ФактическаяСтоимостьПомещения')

filtered_df = filtered_df.dropna(subset=['СтоимостьНаДатуБрони'])
# print(filtered_df.isnull().sum())

# Задание 5
filtered_df['Цена за квадратный мет'] = filtered_df['ФактическаяСтоимостьПомещения'] / filtered_df['ПродаваемаяПлощадь']
filtered_df['Скидка в процентах'] = (filtered_df['СкидкаНаКвартиру'].astype(float) / filtered_df['ФактическаяСтоимостьПомещения']) * 100
# print(filtered_df['Цена за квадратный мет'])
# print(filtered_df['Скидка в процентах'] )
# print(filtered_df['СледующийСтатус'].unique())


# Задание 6
# filtered_df['СкидкаНаКвартиру'] = filtered_df['СкидкаНаКвартиру'].astype(int)
#
#
# def normalize_to_range(series, new_min=0, new_max=1):
#     # Нормализация с заданным диапазоном
#     min_val = series.min()
#     max_val = series.max()
#
#     normalized = (series - min_val) / (max_val - min_val)
#     scaled = normalized * (new_max - new_min) + new_min
#
#     return scaled
#
#
# filtered_df['СкидкаНаКвартиру'] = filtered_df['СкидкаНаКвартиру'].astype(int)
#
#
# def normalize_to_range(series, new_min=0, new_max=1):
#     # Нормализация с заданным диапазоном
#     min_val = series.min()
#     max_val = series.max()
#
#     normalized = (series - min_val) / (max_val - min_val)
#     scaled = normalized * (new_max - new_min) + new_min
#
#     return scaled
#
#
# # Применение
# filtered_df['Тип'] = normalize_to_range(filtered_df['Тип'], new_min=0, new_max=1)
# filtered_df['ПродаваемаяПлощадь'] = normalize_to_range(filtered_df['ПродаваемаяПлощадь'], new_min=0, new_max=1)
# filtered_df['Этаж'] = normalize_to_range(filtered_df['Этаж'], new_min=0, new_max=1)
# filtered_df['СтоимостьНаДатуБрони'] = normalize_to_range(filtered_df['СтоимостьНаДатуБрони'], new_min=0, new_max=1)
# filtered_df['ФактическаяСтоимостьПомещения'] = normalize_to_range(filtered_df['ФактическаяСтоимостьПомещения'],
#                                                                   new_min=0, new_max=1)
# filtered_df['СкидкаНаКвартиру'] = normalize_to_range(filtered_df['СкидкаНаКвартиру'], new_min=-0.5, new_max=0.5)
# filtered_df['Цена за квадратный мет'] = normalize_to_range(filtered_df['Цена за квадратный мет'], new_min=0, new_max=1)
# filtered_df['Скидка в процентах'] = normalize_to_range(filtered_df['Скидка в процентах'], new_min=0, new_max=1)
from sklearn.preprocessing import MinMaxScaler

# Создаем MinMaxScaler
scaler = MinMaxScaler()

# Нормализуем столбцы к диапазону [0, 1]
columns_to_normalize = [
    'Тип',
    'ПродаваемаяПлощадь',
    'Этаж',
    'СтоимостьНаДатуБрони',
    'ФактическаяСтоимостьПомещения',
    'Цена за квадратный мет',
    'Скидка в процентах'
]

filtered_df[columns_to_normalize] = scaler.fit_transform(filtered_df[columns_to_normalize])

# Нормализуем СкидкаНаКвартиру к диапазону [-0.5, 0.5]
scaler_discount = MinMaxScaler(feature_range=(-0.5, 0.5))
filtered_df['СкидкаНаКвартиру'] = scaler_discount.fit_transform(filtered_df[['СкидкаНаКвартиру']])

# print(filtered_df)

# Задание 7
print(filtered_df['СледующийСтатус'].value_counts())

# Задание 8
target = 'СледующийСтатус'
filtered_df = filtered_df[filtered_df['СледующийСтатус'] != 'В резерве']
print(filtered_df['СледующийСтатус'].value_counts())

# Все True и False преобразуем в значение 1 и 0
filtered_df = filtered_df.apply(lambda x: x.astype(int) if x.dtype == bool else x)

filtered_df['СледующийСтатус'] = filtered_df['СледующийСтатус'].astype(int)

# Список факторных признаков
feature_columns = [
    col for col in filtered_df.columns
    if (col != target and
        filtered_df[col].dtype in ['int64', 'float64'])
]

# print(filtered_df[feature_columns])

# Задание 9
X_train, X_test, y_train, y_test = train_test_split(
    filtered_df[feature_columns],
    filtered_df[target],
    test_size=0.2
)

# print(X_train , X_test)

# Задание 10 - 14


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report
)

# 1. KNN Classifier (с параметрами по умолчанию)
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)

# Прогнозы для KNN
knn_train_pred = knn_model.predict(X_train)
knn_test_pred = knn_model.predict(X_test)

# 2. Decision Tree Classifier (с параметрами по умолчанию)
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

# Прогнозы для Decision Tree
dt_train_pred = dt_model.predict(X_train)
dt_test_pred = dt_model.predict(X_test)


# Функция для вычисления метрик
def calculate_metrics(y_true, y_pred, model_name):
    print(f"Метрики для {model_name}:")
    print("F1-мера:", f1_score(y_true, y_pred, average='weighted'))
    print("Precision:", precision_score(y_true, y_pred, average='weighted'))
    print("Recall:", recall_score(y_true, y_pred, average='weighted'))
    print("\n")


# Вычисление метрик для KNN
print("KNN - Обучающая выборка:")
calculate_metrics(y_train, knn_train_pred, "KNN (Train)")

print("KNN - Тестовая выборка:")
calculate_metrics(y_test, knn_test_pred, "KNN (Test)")

# Вычисление метрик для Decision Tree
print("Decision Tree - Обучающая выборка:")
calculate_metrics(y_train, dt_train_pred, "Decision Tree (Train)")

print("Decision Tree - Тестовая выборка:")
calculate_metrics(y_test, dt_test_pred, "Decision Tree (Test)")

# Подробный отчет о классификации
# print("Подробный отчет для KNN:")
# print(classification_report(y_test, knn_test_pred))

# print("\nПодробный отчет для Decision Tree:")
# print(classification_report(y_test, dt_test_pred))


# 15-18
# Построение boxplot для всех числовых признаков
numeric_df = filtered_df.select_dtypes(include=['number']).dropna(subset='СледующийСтатус')
# print(numeric_df)

plt.figure(figsize=(10, 6))
numeric_df.boxplot()
plt.title('Boxplot для всех числовых признаков')
plt.ylabel('Значения')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Функция для удаления выбросов
def remove_outliers_iqr(df):
    # Выбираем только числовые признаки, исключая бинарные
    numeric_df = df.select_dtypes(include=['number'])

    # Фильтруем только те столбцы, которые не являются бинарными
    non_binary_columns = numeric_df.columns[numeric_df.nunique() > 2]

    # Инициализируем DataFrame для хранения очищенных данных
    cleaned_df = df.copy()

    for column in non_binary_columns:
        Q1 = cleaned_df[column].quantile(0.25)
        Q3 = cleaned_df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Фильтруем DataFrame, удаляя выбросы
        cleaned_df = cleaned_df[(cleaned_df[column] >= lower_bound) & (cleaned_df[column] <= upper_bound)]

    return cleaned_df  # возвращаем очищенный DataFrame


cleaned_df = remove_outliers_iqr(filtered_df)
# print(cleaned_df)
plt.figure(figsize=(10, 6))
cleaned_df.boxplot()
plt.title('Boxplot для всех числовых признаков')
plt.ylabel('Значения')
plt.xticks(rotation=45)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    cleaned_df[feature_columns],
    cleaned_df[target],
    test_size=0.2
)

# 1. KNN Classifier (с параметрами по умолчанию)
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)

# Прогнозы для KNN
knn_train_pred = knn_model.predict(X_train)
knn_test_pred = knn_model.predict(X_test)

# 2. Decision Tree Classifier (с параметрами по умолчанию)
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

# Прогнозы для Decision Tree
dt_train_pred = dt_model.predict(X_train)
dt_test_pred = dt_model.predict(X_test)

# Подробный отчет о классификации
print("Подробный отчет для KNN:")
print(classification_report(y_test, knn_test_pred))

print("\nПодробный отчет для Decision Tree:")
print(classification_report(y_test, dt_test_pred))

# LogisticRegression
from sklearn.linear_model import LogisticRegression

logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

logistic_train_pred = logistic_model.predict(X_train)
logistic_test_pred = logistic_model.predict(X_test)

print("Отчет для logistic_model")
calculate_metrics(y_test, logistic_test_pred, "logistic_model")
# print(classification_report(y_test, logistic_test_pred))

# SVM
from sklearn.svm import LinearSVC

svc_model = LinearSVC()
svc_model.fit(X_train, y_train)

svc_train_pred = svc_model.predict(X_train)
svc_test_pred = svc_model.predict(X_test)

print("Отчет для svc model")
calculate_metrics(y_train, svc_train_pred, "svc model")
# print(classification_report(y_test, svc_test_pred))


from sklearn.model_selection import cross_val_score

k_count = range(1, 41)
k_scores = []

for k in k_count:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, scoring='precision')
    k_scores.append(scores.mean())

# Строим график зависимости точности от k
plt.figure(figsize=(8, 6))
plt.plot(k_count, k_scores, marker='o')
plt.xlabel('Количество соседей (k)')
plt.ylabel('параметр')
plt.show()

# Диапазон глубины дерева от 2 до 40
depths = range(2, 41)
depth_scores = []

for depth in depths:
    dt_classifier = DecisionTreeClassifier(max_depth=depth)
    scores = cross_val_score(dt_classifier, X_train, y_train, scoring='accuracy')
    depth_scores.append(scores.mean())

# Строим график зависимости точности от глубины дерева
plt.figure(figsize=(8, 6))
plt.plot(depths, depth_scores, marker='o')
plt.xlabel('Глубина дерева')
plt.ylabel('параметр')
plt.show()

