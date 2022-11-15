# 필요 Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 분석용 데이터 불러오기
dataset = pd.read_csv('source_data.csv')

x = dataset.iloc[:, [15,16]].values
y = dataset.iloc[:, 23].values

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25, random_state=0)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
xtrain = sc_x.fit_transform(xtrain)
xtest = sc_x.transform(xtest)
print(xtrain[0:10, :])

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(xtrain, ytrain)
y_pred = classifier.predict(xtest)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytest, y_pred)
print("혼동행렬 : \n", cm)

from sklearn.metrics import accuracy_score
print("정확도 : ", accuracy_score(ytest, y_pred))

from matplotlib.colors import ListedColormap
x_set, y_set = xtest, ytest
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))

plt.contourf(x1, x2, classifier.predict(
    np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape), alpha=0.75, cmap= ListedColormap(('red','green')))

plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)

plt.title('Classifier')
plt.xlabel('CEWT')
plt.ylabel('CLWT')
plt.legend()
plt.show()