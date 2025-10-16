import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import SVC, LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Вхідний файл
input_file = 'income_data.txt'

# Читання даних
X = []
y = []
count_class1 = 0
count_class2 = 0
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

# Кодування рядкових змінних
label_encoders = []
X_encoded = np.empty(X.shape)

for i, item in enumerate(X[0]):
    if item.replace('.', '', 1).isdigit():  # якщо число (включно з десятковими)
        X_encoded[:, i] = X[:, i].astype(float)
    else:
        le = preprocessing.LabelEncoder()
        X_encoded[:, i] = le.fit_transform(X[:, i])
        label_encoders.append(le)

# Кодування вихідних міток
y_le = preprocessing.LabelEncoder()
y = y_le.fit_transform(y)

X = X_encoded.astype(float)

# Поділ на тренувальні і тестові дані
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

print(f"Початок обчислень...\n")

# Створення SVM-класифікатора
#classifier = OneVsOneClassifier(LinearSVC(random_state=0, max_iter=10000))
classifier = SVC(kernel='poly', degree=2, random_state=0)

classifier.fit(X_train, y_train)

# --- Передбачення для нової точки ---
input_data = ['37', 'Private', '215646', 'HS-grad', '9', 'Never-married',
              'Handlers-cleaners', 'Not-in-family', 'White', 'Male',
              '0', '0', '40', 'United-States']

# Кодування тестової точки
input_data_encoded = []
count = 0
for i, item in enumerate(input_data):
    if item.replace('.', '', 1).isdigit():
        input_data_encoded.append(float(item))
    else:
        le = label_encoders[count]
        if item in le.classes_:
            input_data_encoded.append(le.transform([item])[0])
        else:
            input_data_encoded.append(-1)  # якщо нове значення не зустрічалося
        count += 1

input_data_encoded = np.array(input_data_encoded).reshape(1, -1)

y_test_pred = classifier.predict(X_test)

print("Метрики якості для тестового набору")
# Акуратність (Accuracy)
acc = accuracy_score(y_test, y_test_pred)
print("Accuracy (Акуратність): " + str(round(100 * acc, 2)) + "%")
# Точність (Precision)
prec = precision_score(y_test, y_test_pred, average='weighted')
print("Precision (Точність): " + str(round(100 * prec, 2)) + "%")
# Повнота (Recall)
rec = recall_score(y_test, y_test_pred, average='weighted')
print("Recall (Повнота): " + str(round(100 * rec, 2)) + "%")

f1 = cross_val_score(classifier, X, y, scoring='f1_weighted', cv=3)
print("F1 score: " + str(round(100 * f1.mean(), 2)) + "%")

# Передбачення
predicted_class = classifier.predict(input_data_encoded)
input_data = ['37', 'Private', '215646', 'HS-grad', '9', 'Never-married',
              'Handlers-cleaners', 'Not-in-family', 'White', 'Male', '0', '0', '40', 'United-States']

print(f"Тестова точка: {input_data}")

print("Predicted income class:", y_le.inverse_transform(predicted_class)[0])



