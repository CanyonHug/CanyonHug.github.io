---
title: "밑바닥부터 시작하는 딥러닝 ch.4"
excerpt: "ch.4 신경망 학습"
date: 2022-11-29
categories:
    - AI
tags:
    - DL
use_math: true
last_modified_at: 2022-11-29
---

<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }

    table.dataframe td {
      text-align: center;
      padding: 8px;
    }

    table.dataframe tr:hover {
      background: #b8d1f3; 
    }

    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  </style>
</head>


# 4. 신경망 학습

- 훈련 데이터로부터 weight 최적값을 자동으로 획득하는 **학습** 과정을 배워본다.  
- 신경망 학습 지표인 **손실 함수 loss function**을 배워본다.



## 4.1 데이터에서 학습한다

신경망은 데이터를 통해 학습이 가능하다. 가중치 매개변수 값을 데이터를 통해 결정한다는 뜻이다. 

> 퍼셉트론도 선형 분리 가능 문제라면 데이터로 학습이 가능하다. 선형 분리 가능 문제는 유한 번의 학습을 통해 풀 수 있다는 사실이 **퍼셉트론 수렴 정리 perceptron convergence theorem**으로 증명되어 있다. 하지만 비선형 문제는 (자동으로) 학습할 수 없다. [퍼셉트론 수렴 정리 설명](https://freshrimpsushi.github.io/posts/perceptron-convergence-theorem/)



숫자를 인식하는 알고리즘을 밑바닥부터 설계하는 대신, 이미지에서 **특징 feature**을 추출하고 특징 패턴을 기계학습 기술로 학습하는 방법이 있다. 여기서 특징은 입력 데이터에서 본질적인 데이터를 정확하게 추출할 수 있도록 설계된 변환기를 가리킨다.



이미지 특징은 보통 벡터로 기술하고, 컴퓨터 비전 분야에선 [SIFT, Scale Invariant Feature Transform](https://bskyvision.com/entry/SIFT-Scale-Invariant-Feature-Transform%EC%9D%98-%EC%9B%90%EB%A6%AC), [SURF, Speed-Up Robust Features](https://alex-an0207.tistory.com/165), [HOG, Histogram of Gradient](https://donghwa-kim.github.io/hog.html) 등의 특징을 사용한다. 이런 특징들로 이미지를 벡터로 변환하고, 벡터로 지도 학습 대표 분류 기법인 [SVM, Support Vector Machine](https://bkshin.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-2%EC%84%9C%ED%8F%AC%ED%8A%B8-%EB%B2%A1%ED%84%B0-%EB%A8%B8%EC%8B%A0-SVM), [KNN, K-Neareast Neighbor](https://m.blog.naver.com/bestinall/221760380344) 등으로 학습 가능하다.



다만, 이미지를 벡터로 변환할 때 사용하는 feature는 여전히 사람이 설계하는 것에 주의해야 한다. 즉, 문제에 적합한 특징을 쓰지 않으면 좋은 결과를 얻기 힘들다. 반면, 딥러닝(신경망)은 사람이 feature를 잡아주지 않아도 특징까지도 스스로 학습한다.  



즉, 아래와 같이 세가지 방식이 존재한다.

1. 데이터 $\rightarrow$ 사람이 생각한 알고리즘 $\rightarrow$ 결과  
2. 데이터 $\rightarrow$ 사람이 생각한 특징(SIFT, HOG 등) $\rightarrow$ 기계학습 (SVM, KNN 등) $\rightarrow$ 결과  
3. 데이터 $\rightarrow$ 신경망(딥러닝) $\rightarrow$ 결과

> 딥러닝을 **종단간 기계학습 end-to-end machine learning** 이라고도 한다. 데이터에서 목표 결과를 얻는다는 뜻을 담고 있다.



기계학습에서는 데이터를 **훈련 데이터 training data**와 **시험 데이터 test data**로 나눠 학습과 실험을 수행한다. training data를 통해 최적의 parameter 값을 찾고, test data를 통해 훈련된 모델의 성능을 평가한다. 



training data와 test data를 나누는 이유는 범용적으로 사용 가능한 모델을 원하기 때문이다. 미지의 data에 대해 제대로 작동해야 하기 때문에 test data를 따로 분리하는 것이다. 한 데이터셋에만 지나치게 최적화된 상태를 **오버피팅 overfitting** (과적합, 과학습, 과적응 등)이라고 한다.


## 4.2 손실 함수

신경망 학습은 현재 상태를 하나의 지표로 표현한다. 이 지표를 좋게 만들어주는 parameter 값을 찾는 것이 목표이다. 신경망에 사용하는 지표는 **손실 함수 loss function**이라고 한다. (비용함수 cost function 이라고도 한다.) 손실 함수는 주로 **평균 제곱 오차 MSE, Mean Squared Error**, **교차 엔트로피 오차 CEE, Cross Entropy Error**를 사용한다.



**평균 제곱 오차 MSE, Mean Squared Error**  

MSE는 가장 많이 쓰이는 loss function이다. MSE의 수식은 다음과 같다.  

$$ E = \frac{1}{2}\displaystyle\sum_{k}{(y_k - t_k)^2}$$

$y_k$는 신경망 출력값, $t_k$는 정답 레이블, $k$는 데이터 차원 수를 나타낸다. ch.3 손글씨 인식 데이터를 예로 들면 아래와 같다.  

- y = \[0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0\]  
- t = \[0, 0, 1, 0, 0, 0, 0, 0, 0, 0\]  



(이처럼 한 원소만 1로 하고 그 외는 0으로 하는 표기법을 원-핫 인코딩이라 한다.)



즉, MSE는 각 원소의 추정값과 정답의 차를 제곱한 후, 평균을 구한것이다. 이름 그대로다. 코드로 구현하면 아래와 같다.



```python
import numpy as np

def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)

t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
# '2' 일 확률 높게 예측
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
print(mean_squared_error(np.array(y), np.array(t)))

# '7' 일 확률이 높다고 예측
y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print(mean_squared_error(np.array(y), np.array(t)))
```

<pre>
0.09750000000000003
0.5975
</pre>

***

**교차 엔트로피 오차 CEE, Cross Entropy Error**  

CEE도 loss function으로 많이 사용한다. CEE의 수식은 다음과 같다.  

$$ E = -\sum_{k}{t_k \log y_k} $$  

여기서 $\log$는 밑이 e인 자연로그이다. $t_k$는 정답 레이블로, 원-핫 인코딩이다. 즉, 식은 실질적으로 정답일 때의 추정 ($t_k$가 1일 때의 $y_k$) 의 자연로그를 계산하는 식이다.



```python
import numpy as np
import matplotlib.pylab as plt

x = np.arange(0.1, 1.1, 0.1)
y = -np.log(x)
plt.plot(x, y)
plt.show()
```

<img src="data:image/png;base64,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"/>

-log 의 그래프는 위와 같다. 정답인 원소에 대해 $y_t$ 값이 1에 가까울 수록, loss function 값이 0에 가까워진다. 즉, 정답인 원소를 정답일 확률이 높다고 예측할 수록 loss function 값이 작아진다. CEE는 다음과 같이 구현 가능하다.



```python
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
# '2' 일 확률 높게 예측
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
print(cross_entropy_error(np.array(y), np.array(t)))

# '7' 일 확률이 높다고 예측
y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print(cross_entropy_error(np.array(y), np.array(t)))
```

<pre>
0.510825457099338
2.302584092994546
</pre>

***

np.log()에 0을 입력하면 -inf가 되기 때문에, 작은 값 delta를 더해준다. 



데이터 하나에 대한 loss function을 구해봤으니, training data 전체에 대한 loss function 합을 구해본다. 전체 loss function 값은 다음과 같다.  

$$ E = -\frac{1}{N}\sum_{n}\sum_{k}{t_{nk} \log y_{nk}} $$  

N개의 데이터에 대해 loss function 합을 구하고 N으로 나누어 '평균 손실 함수'를 구한다. 평균을 사용해야 training data 개수와 상관없이 통일된 지표를 얻을 수 있다.



그런데, 데이터셋이 커지면 전체 데이터셋을 한번에 loss function을 계산하는 게 비효율적이다. 이런 경우, training data 중 일부를 골라 학습을 수행한다. (근사치를 활용) 이때 일부를 **미니 배치 mini batch**라고 한다. 아래는 미니배치를 골라내는 코드이다.



```python
import numpy as np
from codes.dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize = \
    True, one_hot_label = True)

print(x_train.shape)
print(t_train.shape)

train_size = x_train.shape[0]
batch_size = 10
# train_size 중 batch_size 만큼 램덤 추출 (60000 중 10개)
batch_mask = np.random.choice(train_size, batch_size)    
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
print(batch_mask)
```

<pre>
(60000, 784)
(60000, 10)
[  223 52692  8405  1553 50390  2020 23663 54992 51601 34865]
</pre>

***

위에서 구한 CEE는 데이터 하나를 대상으로 한 구현으로, 배치 데이터를 처리할 수 있도록 수정해줘야 한다. 데이터 하나인 경우와 배치로 묶인 경우 두가지 모두 처리할 수 있도록 코드를 구현해 준다.



```python
import numpy as np

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t_size)
        y = y.reshape(1, y.size)
        
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y)) / batch_size
```

y가 1차원이면 (데이터 하나당 CEE를 구하는 경우) reshape 함수로 데이터의 형상을 바꿔준다. 그리고 배치 크기로 나눠 정규화하고 이미지 1장당 오차를 계산한다.



label이 원-핫 인코딩이 아니라 '2' 같이 숫자 레이블로 주어질 경우 CEE를 다음과 같이 구할 수 있다.



```python
import numpy as np

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t_size)
        y = y.reshape(1, y.size)
        
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size
```

원-핫 인코딩일 때는 t가 1인 원소의 신경망 출력만으로 CEE를 표시한다. 그래서 t \* np.log(y) 부분을 np.log(y\[np.arange(batch_size), t\])로 바꿔준다. 이렇게 바뀌려면 y\[np.arange(batch_size), t\]가 정답 레이블에 해당하는 신경망의 출력을 의미해야 한다.



np.arange(batch_size)는 0부터 batch_size - 1까지 numpy 배열을 생성하고 t에는 정답 레이블 \[2, 7, 0, 9, 4\]와 같이 저장되어 있다. 따라서 y\[np.arange(batch_size), t\]는 \[y\[0, 2\], y\[1, 7\], y\[2, 0\], y\[3, 9\], y\[4, 4\]\]인 numpy 배열을 생성한다. y가 n * 10이므로, 정답인 숫자를 얼마나 예측했는지에 대한 결과값을 의미한다.



지금까지 loss function을 알아봤다. 그런데 왜 loss function을 정의하는가? 정확도를 이용하면 되지 않을까 의문이 들 수 있다. 신경망은 학습 과정에서 미분을 통해 parameter 값을 변화시킬 때 결과값이 얼마나 변화하는지 알아보면서 조정한다. (parameter 미분 값이 음수면 parameter를 양수 방향으로 움직여 loss function 값을 줄인다. 미분 값이 0이 되면 parameter 갱신을 멈춘다.) 



정확도는 미분 값이 대부분의 장소에서 parameter 미분 값이 0이 되기 때문에 parameter를 갱신하기 어렵다. (매개변수를 약간 조정하더라도 정확도는 그대로 유지될 확률이 높다. 또한, 정확도가 개선되어도 불연속적인 값을 가질 것이다.) 반면, loss function은 parameter 값에 따라 연속적으로 변화하기 때문에 정확도 대신 loss function을 학습 지표로 사용하는 것이다.



이와 비슷한 이유로 activation function으로 step function을 사용하지 않는다. step function의 미분은 0이외의 곳에서 모두 0이기 때문에 loss function을 지표로 삼는게 의미가 없어진다. (sigmoid는 반대로 미분 값이 항상 0이 아니고, 연속적이라 사용된다.)


## 4.3 수치 미분

경사법은 기울기 값을 기준으로 나아갈 방향을 정한다. 기울기 값은 미분을 통해 구할 수 있다. 미분은 순간의 변화량을 나타낸 것으로 다음의 수식으로 나타낼 수 있다.  

$$ \frac{df(x)}{dx} = \displaystyle\lim_{h \rightarrow 0} {\frac{f(x + h) - f(x)} {h}} $$  

위의 식으로 파이썬으로 함수의 미분을 구현하면 아래와 같다.



```python
def numerical_diff(f, x):
    h = 10e-50
    return (f(x + h) - f(x)) / h
```

$\displaystyle\lim_{h \rightarrow 0}$를 파이썬에서 근사하기 위해 h에 최대한 작은 값을 대입해서 미분 값을 구해준다. 이처럼 작은 값 사이에 기울기로 미분값을 근사하는 것을 **수치 미분 numerical differentiation**이라고 한다. (수식을 전개해 미분을 구하는 것을 **해석적 analytic**이라고 한다.) 하지만 여기엔 2가지 문제가 있다.  

1. h에 10e-50이라는 숫자는 **반올림 오차 rounding error**를 일으킨다.  
2. $\displaystyle\lim_{h \rightarrow 0}$ 대신 h에 작은 값을 넣어 구하기 때문에 오차가 존재한다.  

- 1번 문제는 너무 작은 값을 저장하게 되면, 메모리가 모자라서 올바르게 표현되지 않고 0으로 표현되기 때문에 나타나는 문제다. $10^{-50}$ 대신 $10^{-4}$ 정도의 값을 사용하면 좋은 결과를 얻는다고 알려져 있다.  
- 2번 문제는 x 점에서의 f에 대한 접선의 기울기 대신 x와 x + h 사이의 f의 기울기를 구하는 데서 오는 오차다. 이를 줄이기 위해 (x + h)와 (x - h) 사이의 기울기를 구한다. 이를 x 중심으로 전후의 차분(임의 두점에서 함수 값 차이)를 계산한다는 의미에서 **중심 차분, 중앙 차분 central difference**라고 한다. [중앙 차분 설명](https://blog.naver.com/mykepzzang/220072089756)



위의 두 개선점으로 numerical differentiation을 다시 구현한다.



```python
def numerical_diff(f, x):
    h = 10e-4
    return (f(x + h) - f(x - h)) / (2 * h)
```

만약, 함수에 변수가 여러 개 포함되면, 어느 변수에 대한 미분인지가 중요하다. 변수가 여럿일 때 한 변수에 대한 미분을 **편미분 partial derrivative**라고 한다. 다른 변수를 상수로 보고 한 변수를 변화시킬 때 변화량을 관찰하는 것이다. (다변수함수에서 모든 변수의 변화에 따라 변화하는 행태를 근사하는 양은 **전미분 total derivative**라고 한다.)  편미분은 $\frac{\partial f}{\partial x_0}$ 로 쓴다.


## 4.4 기울기
**기울기 gradient**는 아래와 같이 구할 수 있다.  

```python
import numpy as np

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = x[idx]

        # f(x + h) 계산
        x[idx] = tmp_val + h
        fxh1 = f(x)
        
        # f(x - h) 계산
        x[idx] = tmp_val - h
        fxh2 = f(x)
        
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val    # 값 복원
        
    return grad
```

반복문을 통해 각 변수에 대해 편미분을 해준 것을 알 수 있다. np.zeros_like(x는 x와 형상이 같고 원소가 모두 0인 배열을 만든다. 다음의 예시를 통해 $x_0^2 + x_1^2$ 함수에 대해 gradient를 구해본다.



```python
def function_2(x):
    return x[0] ** 2 + x[1] ** 2

print(numerical_gradient(function_2, np.array([3.0, 4.0])))
print(numerical_gradient(function_2, np.array([3.0, 0.0])))
```

<pre>
[6. 8.]
[6. 0.]
</pre>
> 기울기 gradient는 벡터 미적분학에서 스칼라장의 최대의 증가율을 나타내는 벡터장을 뜻한다.



즉, $x_0^2 + x_1^2$에 대하여 gradient는 각 위치지점에서 어느 방향으로 이동해야 함수 값이 최대로 증가하는지 알려준다.



이 개념을 활용해 기계학습에서 최적의 parameter를 찾기 위해 경사 하강법을 사용한다. gradient 방향의 반대로 이동하여 손실 함수 값을 낮추는 것을 경사 하강법이라고 한다. 일반적인 문제의 loss function은 매우 복잡하기 때문에 경사 하강법으로 국소적인 탐색을 시행하는 것이다. 그렇기 때문에 기울기가 0인 최솟값이 아닌 극솟값, 안장점에 빠질 수 있다.  

> 경사법은 최솟값, 최댓값을 찾는지에 따라 **경사 하강법 gradient descent method**, **경사 상승법 gradient ascent method**라고 한다. loss function의 부호를 반전시키면 최솟값을 찾는 문제가 최댓값을 찾는 문제로 바뀌어 본질적으로 중요치 않다. 일반적으로 신경망 분야에서 경사법은 경사 하강법으로 등장할 때가 많다.



위의 $x_0^2 + x_1^2$ 함수에 대해 경사하강법을 수식으로 나타내면 아래와 같다.  

$$ x_0 = x_0 - \eta \frac{\partial f}{\partial x_0} $$

$$ x_1 = x_1 - \eta \frac{\partial f}{\partial x_1} $$

변수가 늘어도 각 변수 편미분을 통해 업데이트 방식은 동일하다. $\eta$는 신경망 학습에서 **학습률 learning rate**를 의미한다. 0.01 같이 특정 값으로 미리 정해둬야 한다. 너무 크거나 작으면 좋은 장소를 찾아갈 수 없다. 보통 이 learning rate 값을 변경하면서 올바르게 학습하고 있는지 확인하며 진행한다. 경사하강법은 다음의 코드로 구현 가능하다.



```python
def gradient_descent(f, init_x, lr = 0.01, step_num = 100):
    x = init_x
    
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x
```

위의 gradient descent 함수를 활용해 $x_0^2 + x_1^2$의 최솟값을 찾으면 다음과 같다.



```python
def function_2(x):
    return x[0] ** 2 + x[1] ** 2

init_x = np.array([-3.0, 4.0])
result = gradient_descent(function_2, init_x = init_x, lr = 0.1, step_num = 100)
print(result)
print(function_2(result))
```

<pre>
[-6.11110793e-10  8.14814391e-10]
1.0373788922158197e-18
</pre>

***

이 방식을 그림으로 나타내면 아래와 같다.



```python
import numpy as np
import matplotlib.pylab as plt


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append( x.copy() )

        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x, np.array(x_history)


def function_2(x):
    return x[0]**2 + x[1]**2

init_x = np.array([-3.0, 4.0])    

lr = 0.1
step_num = 20
x, x_history = gradient_descent(function_2, init_x, lr=lr, step_num=step_num)

plt.plot( [-5, 5], [0,0], '--b')
plt.plot( [0,0], [-5, 5], '--b')
plt.plot(x_history[:,0], x_history[:,1], 'o')

plt.xlim(-3.5, 3.5)
plt.ylim(-4.5, 4.5)
plt.xlabel("X0")
plt.ylabel("X1")
plt.show()
```

<img src="data:image/png;base64,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"/>

> learning rate 같은 parameter를 **하이퍼 파라미터 hyper parameter**라고 한다. weight와 bias 같이 training으로 값이 결정되는 대신, hyper parameter는 사람이 직접 설정해야 한다.일반적으로 여러 후보 값 중에서 시험을 통해 가장 잘 학습하는 값을 찾는 과정을 거쳐야 한다.



학습을 위해서 기울기를 구해야 한다. 여기서 기울기는 weight parameter에 관한 lsos function의 기울기이다. 가중치가 $W$, loss function이 $L$인 신경망의 경우, 경사를 $\frac {\partial L} {\partial W}$로 나타낼 수 있다.



weight가 2 \* 3인 신경망을 코드로 구현해 본다.



```python
import numpy as np
from codes.common.functions import softmax, cross_entropy_error
from codes.common.gradient import numerical_gradient


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3) # 정규분포로 초기화

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss

x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])

net = simpleNet()

f = lambda w: net.loss(x, t)
dW = numerical_gradient(f, net.W)

print(dW)
```

<pre>
[[ 0.26231427  0.31891381 -0.58122807]
 [ 0.3934714   0.47837071 -0.87184211]]
</pre>

***

신경망 기울기를 구한 다음에는 경사법 따라 weight parameter를 갱신하면 된다.  



## 4.5 학습 알고리즘 구현하기

전제 : 신경망에는 적응 가능한 weight, bias가 있고, 이를 데이터에 조정시키는 과정을 학습이라 한다. 학습은 아래 4단계로 구성된다.  

1. 미니배치  
훈련 데이터 중 일부 무작위로 가져온 것을 mini-batch라 하며, mini-batch의 loss function 값을 줄이는 것을 목표로 한다.  
2. 기울기 산출  
mini-batch의 loss function 값을 줄이기 위해 각 weight의 기울기를 구한다.  
3. parameter 갱신  
weight를 기울기 반대 방향으로 갱신한다.  
4. 반복  
1~3 단계를 반복한다.



이는 경사 하강법으로 parameter를 갱신하는 방법이며, 데이터를 mini-batch로 무작위 선정하기 때문에 **확률적 경사 하강법 stochastic gradient descent, SGD**라고 부른다.



실제로 손글씨 학습 신경망을 구현해본다. 2층 신경망(은닉층이 1개인 네트워크)를 대상으로 MNIST 데이터셋을 사용하여 학습을 수행한다.



```python
from codes.common.functions import *
from codes.common.gradient import numerical_gradient


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
    
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        return y
        
    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t):
        y = self.predict(x)
        
        return cross_entropy_error(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    # x : 입력 데이터, t : 정답 레이블
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
        
    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}
        
        batch_num = x.shape[0]
        
        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        
        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads
```

gradient()외에 나머지는 위에서 봤던 내용과 유사하다. gradient()는 오차역전파법을 사용해 기울기를 효율적이고, 빠르게 계산한다. 추후에 오차역전파를 알아본다. numerical_gradient()는 수치 미분 방식으로 parameter 기울기를 계산하기 때문에 gradient()를 사용하는게 더 빠르다.



TwoLayerNet 클래스와 MNIST 데이터셋을 사용해 mini-batch 학습을 구현해 본다.



```python
import numpy as np
import matplotlib.pyplot as plt
from codes.dataset.mnist import load_mnist

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 하이퍼파라미터
iters_num = 10000  # 반복 횟수를 적절히 설정한다.
train_size = x_train.shape[0]
batch_size = 100   # 미니배치 크기
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

# 1에폭당 반복 수 : 60000 / 100 = 600
iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    # 미니배치 획득
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 기울기 계산
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)
    
    # 매개변수 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    # 학습 경과 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    # 1에폭당 정확도 계산
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

# 그래프 그리기
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
```

<pre>
train acc, test acc | 0.09006666666666667, 0.0919
train acc, test acc | 0.7701, 0.7737
train acc, test acc | 0.8751, 0.8782
train acc, test acc | 0.8977666666666667, 0.8998
train acc, test acc | 0.9070666666666667, 0.9096
train acc, test acc | 0.9140666666666667, 0.9161
train acc, test acc | 0.91995, 0.9215
train acc, test acc | 0.9235666666666666, 0.9246
train acc, test acc | 0.9275833333333333, 0.93
train acc, test acc | 0.9317666666666666, 0.9323
train acc, test acc | 0.9339666666666666, 0.9354
train acc, test acc | 0.9369166666666666, 0.9375
train acc, test acc | 0.9396666666666667, 0.9389
train acc, test acc | 0.9420666666666667, 0.9409
train acc, test acc | 0.9435, 0.9418
train acc, test acc | 0.9453166666666667, 0.9446
train acc, test acc | 0.9469166666666666, 0.9454
</pre>
<img src="data:image/png;base64,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"/>

***

**에폭 epoch**은 하나의 단위로, 1 epoch은 훈련 데이터를 모두 소진했을 때의 횟수에 해당한다. 데이터 10,000개를 100개의 mini-batch로 학습할 경우, SGD를 100번 반복하면 모든 훈련 데이터를 소진한 것이 된다. 이 경우, 100회가 1 epoch이다.

위의 상황에서는 전체 데이터 사이즈가 60,000이고, batch size가 100 이므로, 1 epoch당 반복 횟수가 600이다. 즉 batch size * iter_per_epoch = 1 epoch 이다. 코드에서 iters_num은 mini-batch를 뽑아 학습하는 과정을 10,000번 진행한다는 뜻이다. 10,000 / 600 = 16.6~ epoch만큼 학습을 진행한다. [batch, iter, epoch 설명](https://jonhyuk0922.tistory.com/129)

1 epoch 당 training, test data에 대한 정확도를 보여준다. training, test data에 대한 정확도를 나타낸 선이 거의 겹쳐 있으므로, over-fitting이 일어나지 않았음을 알 수 있다.  

> over-fitting이 일어나면, 어느 순간 test data에 대한 정확도가 점차 떨어지기 시작한다. 이를 포착해 학습을 중단하면 over-fitting을 예방할 수 있다. 이를 **조기 종료 early stopping**라고 하며 drop-out 같은 방식이 있다.


## 4.6 정리
- machine learning에 쓰이는 data set은 training data, test data로 나눠 사용한다.
- training 한 모델의 범용 능력을 test data로 평가한다.  
- 신경망 학습은 loss function을 지표로, loss function의 값이 작아지는 방향으로 parameter를 업데이트한다.  
- weight 갱신할 시, gradient를 이용하여 기울기 반대 방향으로 이동해 loss function 값을 줄인다. (경사 하강법)  
- 아주 작은 값의 차분으로 미분하는 것을 수치 미분이라고 한다.
- 수치 미분은 구현이 간단하지만, 시간이 오래걸린다. 오차역전파는 구현이 복잡하지만, 빠르게 gradient를 구할 수 있다.