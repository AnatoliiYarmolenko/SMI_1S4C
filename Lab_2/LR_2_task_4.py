import numpy as np
import warnings
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.exceptions import ConvergenceWarning

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC 

input_file = 'income_data.txt'
X = []
y = []
count_class1 = 0
count_class2 = 0
# Обмежимо набір даних для швидшого виконання та балансування
max_datapoints = 25000 

with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:
            continue

        data = line.strip().split(', ')
        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data[:-1])
            y.append(data[-1])
            count_class1 += 1
        elif data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data[:-1])
            y.append(data[-1])
            count_class2 += 1

# Перетворення у numpy-масиви
X = np.array(X)
y = np.array(y)

# Кодування рядкових змінних (ознак)
X_encoded = np.empty(X.shape)
for i, item in enumerate(X[0]):
    if item.replace('.', '', 1).isdigit():
        X_encoded[:, i] = X[:, i].astype(float)
    else:
        le = preprocessing.LabelEncoder()
        X_encoded[:, i] = le.fit_transform(X[:, i])

# Кодування вихідних міток (класів)
y_le = preprocessing.LabelEncoder()
y = y_le.fit_transform(y)

X = X_encoded.astype(float)

print(f"Дані завантажено та оброблено.")
print(f"Розмір набору ознак (X): {X.shape}")
print(f"Розмір міток (y): {y.shape}")
print("-" * 40)


# Створюємо список моделей для тестування
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', LinearSVC(random_state=0, max_iter=5000, dual=False))) # dual=False краще, коли n_samples > n_features

# Оцінка кожної моделі
results = []
names = []
scoring = 'accuracy' # Метрика якості - точність

print("Оцінка моделей (10-кратна крос-валідація):")
print(f"Метрика: {scoring}")
print("Формат: Назва: Cереднє (Стандартне відхилення)")
print("-" * 40)

# Ігноруємо попередження про збіжність (для LR та SVM, які можуть
# потребувати більше ітерацій, але для порівняння це не критично)
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=UserWarning)

for name, model in models:
    # 10-кратна стратифікована крос-валідація
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    
    # Обчислюємо якість
    cv_results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
    
    results.append(cv_results)
    names.append(name)
    print(f'{name}: {cv_results.mean():.4f} ({cv_results.std():.4f})')

print("-" * 40)
print("Порівняння завершено.")