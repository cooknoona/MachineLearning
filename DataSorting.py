# 데이터 전처리 : 머신러닝 모델의 성능 향상을 위한 중요 과정, 특히 특성의 스케일이 다를 경우 처리가 필요
import numpy as np

from MarketAndMachineLearning import fish_target
from TrainingSetAndTestSet import train_input, test_input, train_target, test_target

fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8,
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7,
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

# 두 리스트를 넘파이의 column_stack() 함수를 이용해 하나의 배열로 병합 (길이와 무게 데이터를 합침)
fish_data = np.column_stack((fish_length, fish_weight))
fish_target = np.concatenate((np.ones(35), np.zeros(14)))

# 사이킷런으로 훈련세트와 테스트세트 나누기
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = (
    train_test_split(fish_data, fish_target, random_state=42)
)

# 모델 훈련 : 스케일 차이로 인한 오류 발생
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()
# kn.fit(train_input, train_target)
# score = kn.score(test_input, test_target)
# print(f"모델 학습 결과 : {score:.2f}")

# 표준화 적용 : 표준 편차
mean = np.mean(train_input, axis = 0)
std = np.std(train_input, axis = 0)
train_scaled = (train_input - mean) / std
test_scaled = (test_input - mean) / std
kn.fit(train_scaled, train_target)
new_sample = ([25, 150] - mean) / std
print(kn.predict([new_sample]))

# 임의의 테스트 데이터 입력
# print(kn.predict([[25, 150]]))  # 길이가 25cm, 무게가 150g, 실제로는 도미

# 시각화
# import matplotlib.pyplot as plt
# plt.scatter(train_input[:,0], train_input[:,1]) # 앞의 매개변수 x 축, 뒤의 매개변수는 y 축
# plt.scatter(25, 150, marker='^')
# plt.xlabel('length')
# plt.ylabel('weight')
# plt.show()

