import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
       21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
       23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
       27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
       39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
       44.0])
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

# 데이터 분리
train_input, test_input, train_target, test_target = train_test_split(
    perch_length, perch_weight, random_state=42)
train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)

# k-최근접 이웃 회귀 모델
knr = KNeighborsRegressor(n_neighbors=3)
knr.fit(train_input, train_target)

# 선형 회귀 모델
lr = LinearRegression()
lr.fit(train_input, train_target)

# 다항 회귀 모델
train_poly = np.column_stack((train_input ** 2, train_input))
test_poly = np.column_stack((test_input ** 2, test_input))
lr_poly = LinearRegression()
lr_poly.fit(train_poly, train_target)

# 예측 및 결과
print("k-최근접 이웃 예측 (50cm):", knr.predict([[50]]))
print("선형 회귀 예측 (50cm):", lr.predict([[50]]))
print("다항 회귀 예측 (50cm):", lr_poly.predict([[50**2, 50]]))

# 점수 출력
print("k-최근접 이웃 R2:", knr.score(test_input, test_target))
print("선형 회귀 R2:", lr.score(test_input, test_target))
print("다항 회귀 R2:", lr_poly.score(test_poly, test_target))

# 그래프
plt.figure(figsize=(10, 5))
point = np.arange(15, 50)

# k-최근접 이웃
plt.scatter(train_input, train_target, label='Train Data')
plt.scatter(50, knr.predict([[50]]), marker='^', label='k-NN (50cm)', color='orange')

# 선형 회귀
plt.plot([15, 50], [15 * lr.coef_ + lr.intercept_, 50 * lr.coef_ + lr.intercept_], label='Linear Regression')

# 다항 회귀
plt.plot(point, 1.01 * point**2 - 21.6 * point + 116.05, label='Polynomial Regression')

plt.xlabel('Length (cm)')
plt.ylabel('Weight (g)')
plt.legend()
plt.show()