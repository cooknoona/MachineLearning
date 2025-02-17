# 지도 학습 : 입력 데이터와 그에 해당하는 정답을 사용해 모델을 학습하는 방법.
# 학습 데이터에서 입력과 정답의 관계를 학습하여 샐운 입력에 대해 정확히 예측 할 수 있도록 하는 것
# 훈련 세트 : 모델 학습용 데이터
# 테스트 세트 : 모델 평가용 데이터
# 즉, 같은 데이터로 학습과 평가를 진행하면 모델 데이터를 학습 해버렸기 때문에 정확한 평가 불가
import numpy as np
from sklearn.neighbors import KNeighborsClassifier  # KNN 라이브러리
import matplotlib.pyplot as plt # 시각화 라이브러리
from sphinx.addnodes import index

from MarketAndMachineLearning import fish_target, prediction

# 생선의 길이와 무게 데이터
given_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
given_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

# 생선 데이터와 타깃 데이터 준비
fish_data = [[l, w] for l, w in zip(given_length, given_weight)]
fish_target = [1] * 35 + [0] * 14  # 도미는 1, 빙어는 0

# 넘파이 배열로 변환
input_arr = np.array(fish_data)
target_arr = np.array(fish_target)

# 데이터 섞기
np.random.seed(42)  # 일정한 결과를 얻기 위해 랜덤 시드 설정
index = np.arange(49)  # 0부터 48까지 인덱스 생성
np.random.shuffle(index)  # 인덱스 섞기

# 훈련 세트와 테스트 세트 나누기
train_input = input_arr[index[:35]]  # 훈련 입력 데이터
train_target = target_arr[index[:35]]  # 훈련 타깃 데이터
test_input = input_arr[index[35:]]  # 테스트 입력 데이터
test_target = target_arr[index[35:]]  # 테스트 타깃 데이터

# 훈련 데이터와 테스트 데이터 시각화
plt.scatter(train_input[:, 0], train_input[:, 1], label='train set')
plt.scatter(test_input[:, 0], test_input[:, 1], label='test set')
plt.xlabel('length')
plt.ylabel('weight')
plt.legend()
plt.show()

# 모델 훈련 및 평가
kn = KNeighborsClassifier()
kn.fit(train_input, train_target)  # 모델 학습

# 모델 평가 및 결과 확인
train_score = kn.score(train_input, train_target)  # 훈련 세트 정확도
test_score = kn.score(test_input, test_target)  # 테스트 세트 정확도
print("훈련 세트 정확도:", train_score)
print("테스트 세트 정확도:", test_score)

# 모델의 훈련 및 테스트 세트에 대한 정확도를 비교해 일반화 성능 확인
if train_score > test_score:
    print("모델이 훈련 세트에 과적합(overfitting)되었습니다.")
elif train_score < test_score:
    print("모델이 훈련 세트를 충분히 학습하지 못했습니다(underfitting).")
else:
    print("모델이 훈련 세트와 테스트 세트에서 균형 잡힌 성능을 보입니다.")

# 테스트 세트 예측 결과 확인
predictions = kn.predict(test_input)
print("예측 결과:", predictions)
print("실제 타깃:", test_target)