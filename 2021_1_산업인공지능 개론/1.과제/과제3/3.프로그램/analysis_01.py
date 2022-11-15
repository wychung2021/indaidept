# 필요 Library
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

# 분석용 데이터 불러오기
dataset = pd.read_csv('source_data.csv')
val = dataset.iloc[:, [1, 15]].values

data = np.array(val)
plt.scatter(data[:, 0], data[:, 1])
plt.title("Linear Regression")
plt.xlabel("TIME")
plt.ylabel("CEWT")
plt.axis("equal")

x1 = data[:, 0].reshape(-1, 1)
y1 = data[:, 1].reshape(-1, 1)

# Linear Regression 모델 학습
model = LinearRegression()
model.fit(x1, y1)

# Linear Regression 예측값 계산
y_pred = model.predict(x1)
plt.plot(x1, y_pred, color='r')
plt.show()