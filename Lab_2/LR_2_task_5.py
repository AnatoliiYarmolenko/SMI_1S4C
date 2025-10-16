import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split # Додано імпорт
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from io import BytesIO # neded for plot
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

# ===================================================
# Приклад класифікатора Ridge
# ======================================================================

# Завантаження даних
iris = load_iris()
X, y = iris.data, iris.target

# Розділення даних на навчальну та тестову вибірки
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Створення та навчання класифікатора Ridge
clf = RidgeClassifier(tol = 1e-2, solver = "sag")
clf.fit(Xtrain, ytrain)

# Отримання прогнозу на тестових даних
ypred = clf.predict(Xtest) # X_test -> Xtest

# --- Розрахунок показників якості ---
print('Accuracy:', np.round(metrics.accuracy_score(ytest, ypred), 4))
print('Precision:', np.round(metrics.precision_score(ytest, ypred, average = 'weighted'), 4))
print('Recall:', np.round(metrics.recall_score(ytest, ypred, average = 'weighted'), 4))
print('F1 Score:', np.round(metrics.f1_score(ytest, ypred, average = 'weighted'), 4))
print('Cohen Kappa Score:', np.round(metrics.cohen_kappa_score(ytest, ypred), 4))
print('Matthews Corrcoef:', np.round(metrics.matthews_corrcoef(ytest, ypred), 4))

# Детальний звіт по класах
print('\t\tClassification Report:\n',
      metrics.classification_report(ytest, ypred)) #ypred, ytest -> ytest, ypred

# --- Побудова та збереження матриці плутанини ---
mat = confusion_matrix(ytest, ypred)
sns.heatmap(mat.T, square = True, annot = True, fmt = 'd', cbar = False,
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('True Label (Справжня мітка)')
plt.ylabel('Predicted Label (Прогнозована мітка)');
plt.savefig("Confusion.jpg")

print("\nМатрицю плутанини збережено у файл 'Confusion.jpg'")

# Збереження SVG у об'єкт в пам'яті (як було в оригінальному коді)
f = BytesIO()
plt.savefig(f, format = "svg")

