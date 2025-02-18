# 다중 회귀 (Multiple Linear Regression) : 여러 개의 입력 특성 (독립 변수)를 사용해
# 하나의 타겟 값 (종적 변수) 를 예측하는 선형 회귀 모델

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import matplotlib.pyplot as plt

# 데이터 불러오기 및 준비
df = pd.read_csv('https://bit.ly/perch_csv_data')
perch_full = df.to_numpy()
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0, 115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0, 150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0, 218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0, 556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0, 850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0, 1000.0])

train_input, test_input, train_target, test_target = train_test_split(perch_full, perch_weight, random_state=42)

# 다항 특성 생성
poly = PolynomialFeatures(degree=5, include_bias=False)
train_poly = poly.fit_transform(train_input)
test_poly = poly.transform(test_input)

# 데이터 표준화
ss = StandardScaler()
train_scaled = ss.fit_transform(train_poly)
test_scaled = ss.transform(test_poly)

# 릿지 회귀 적용
ridge = Ridge(alpha=0.1)
ridge.fit(train_scaled, train_target)
print('Ridge Train Score:', ridge.score(train_scaled, train_target))
print('Ridge Test Score:', ridge.score(test_scaled, test_target))

# 라쏘 회귀 적용
lasso = Lasso(alpha=10)
lasso.fit(train_scaled, train_target)
print('Lasso Train Score:', lasso.score(train_scaled, train_target))
print('Lasso Test Score:', lasso.score(test_scaled, test_target))
print('Lasso zero coefficients:', np.sum(lasso.coef_ == 0))