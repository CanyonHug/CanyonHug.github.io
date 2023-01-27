---
title: "밑바닥부터 시작하는 딥러닝 ch.5"
excerpt: "ch.5 오차역전파법"
date: 2022-11-30
categories:
    - AI
tags:
    - DL
use_math: true
last_modified_at: 2022-11-30
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


# 5. 오차역전파법

- 수치미분법보다 빠르게 기울기를 계산하는 오차역전파법을 알아본다.



오차역전파법은 수식, 계산 그래프를 통해 이해할 수 있다. 이 중 계산 그래프를 통해 시각적으로 이해해본다. 계산 그래프는 스탠포드 대학교의 CS231n 수업을 참고했다고 한다.



## 5.1 계산 그래프

**계산 그래프 computational graph**는 계산 과정을 그래프로 나타낸 것이다. 복수의 **노드 node**와 노드 사이의 화살표 **엣지 edge**로 표현된다. 계산그래프를 이용한 문제 풀이는  

1. 계산 그래프를 구성한다.  
2. 그래프에서 계산을 왼쪽에서 오른쪽으로 진행한다.  

의 순서로 진행된다. 특히 2번을 **순전파 forward propagation**라고 한다. 반대 방향으로 가면 **역전파 backward propagation**라고 한다.



계산 그래프에 관한 설명은 [이 분 블로그](https://minding-deep-learning.tistory.com/21)로 대체한다.



계산 그래프는 '국소적 계산'을 전파함으로써 최종 결과를 얻는다. 각 노드에서의 계산은 국소적으로 이뤄진다. (자동차 조립 라인과 유사하다.) 이 때문에 back-propagation으로 미분을 효율적으로 계산 가능하다. 


## 5.2 연쇄법칙

back-propagation은 오른쪽에서 왼쪽으로 국소적인 미분을 전달한다. 이는 **연쇄 법칙 chain rule**에 따른 것이다. chain rule은 **합성 함수**의 미분에 관한 성질이다. 합성 함수는 여러 함수로 구성된 함수로, 예를 들어 $z = (x + y)^2$는 아래 두 식으로 구성된다.  

$$ z = t^2 $$

$$ t = x + y $$  

> 이때 합성 함수의 미분은 합성 함수를 구성하는 각 함수의 미분의 곱으로 나타낼 수 있다.



예를 들면 $\frac {\partial z} {\partial x}$는 $\frac {\partial z} {\partial t}$와 $\frac {\partial t} {\partial x}$의 곱으로 나타내어진다는 것이다.  

$$ \frac {\partial z} {\partial x} = \frac {\partial z} {\partial t} \frac {\partial t} {\partial x} = 2t * 1 = 2(x + y)$$  

이 chain rule 에서 각 함수의 미분의 곱을 역전파에서는 오른쪽에서 왼쪽으로 가면서 각 함수의 미분을 곱해주는 것이다.


## 5.3 역전파

- 덧셈 노드의 역전파  

$z = x + y$를 통해 덧셈 노드에서의 역전파를 알아본다. 각 변수에 대한 편미분을 구해본다.  

$$ \frac {\partial z} {\partial x} = 1 $$

$$ \frac {\partial z} {\partial y} = 1 $$  

x와 y에 대한 함수의 편미분이 모두 1이다. 따라서 덧셈 노드는 역전파로 흘려보낼 때 입력값을 그대로 흘려보낸다. (1을 곱해주기 때문)  

- 곱셈 노드의 역전파  

$z = xy$를 통해 덧셈 노드에서의 역전파를 알아본다. 각 변수에 대한 편미분을 구해본다.  

$$ \frac {\partial z} {\partial x} = y $$

$$ \frac {\partial z} {\partial y} = x $$  

곱셈 노드 역전파는 상류 값에 순전파 때 입력 신호들을 '서로 바꾼 값'을 곱해 하류로 보낸다. (덧셈 역전파에선 상류 값을 그대로 보내서 forward-propagation 입력 신호가 필요치 않지만, 곱셈 back-propagation은 forward 방향의 입력 신호 값이 필요하다.)


## 5.4 단순한 계층 구현하기

위에서 배운 내용으로 단순한 계층을 구현해본다. 각 계층은 forward()와 backward()라는 공통 method를 갖도록 구현한다.



- 곱셈 계층  



```python
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y                
        out = x * y

        return out

    def backward(self, dout):
        dx = dout * self.y  # x와 y를 바꾼다.
        dy = dout * self.x

        return dx, dy
```

역전파의 출력 순서를 잘 확인한다.



- 덧셈 계층



```python
class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y

        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1

        return dx, dy
```

덧셈 계층은 back-propagation에서 순전파 방향 입력 신호를 활용하지 않기 때문에 \_\_init\_\_에 변수를 초기화해주지 않아도 된다.



필요한 계층을 만들어 순전파 method인 forward()를 순서대로 호출하고, 순전파와 반대 순서로 backward()를 호출하면 원하는 미분값을 구할 수 있다.


## 5.5 활성화 함수 계층 구현하기

신경망 구성 층 각각을 클래스 하나로 구현한다. activation function인 ReLU와 Sigmoid 계층을 구현한다.  

- ReLu 계층  

ReLU의 식과 미분은 다음과 같다.  

$$ y = \begin{cases} x \quad (x \gt 0) \\ 0 \quad (x \leq 0) \end{cases}$$

$$ \frac {\partial y} {\partial x} = \begin{cases} 1 \quad (x \gt 0) \\ 0 \quad (x \leq 0) \end{cases}$$

순전파 때의 입력인 x가 0보다 크면, 상류 값을 그대로 흘려주고, x가 0 이하면 하류로 신호를 보내지 않는다.



```python
class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx
```

Relu 클래스는 mask라는 인스턴트 변수를 가지는데, mask는 True/False로 구성된 numpy 배열이다. mask 변수를 통해 0을 씌워 신호를 차단시킨다.



- Sigmoid 계층  

sigmoid의 식은 다음과 같다.  

$$ y = \frac {1} {1 + \exp (-x)} $$  

sigmoid의 계산 그래프 순서는 다음과 같다.  



1. x, -1 $\rightarrow$ (* node) $\rightarrow$ -x  
2. -x $\rightarrow$ (exp node) $\rightarrow$ exp(-x)  
3. exp(-x), 1 $\rightarrow$ (+ node) $\rightarrow$ 1 + exp(-x)  
4. 1 + exp(-x) $\rightarrow$ (/ node) $\rightarrow$ $\frac {1} {1 + exp(-x)}$  



위의 sigmoid 그래프를 역전파 흐름을 따라가본다.  

1. '/' 노드, $ y = \frac {1} {x}$을 미분하면 다음과 같다.  
$$ \begin{aligned} 
\frac {\partial y} {\partial x} &= -\frac {1} {x^2} \\
&= -y^2
\end{aligned} $$  
역전파 때는 상류의 예측값에 $-y^2$을 해서 하류로 전달한다.  
2. '+' 노드, 상류값을 그대로 하류로 흘려보낸다.  
3. 'exp' 노드는 y = exp(x) 연산으로, 미분은 아래와 같다.  
$$ \frac {\partial y} {\partial x} = \exp(x) = y$$  
역전파 때 상류의 값에 순전파 때의 출력(y)를 곱해 하류로 전달한다.  
4. '\*' 노드는 순전파 때 값을 서로 바꿔 곱한다. 여기선 -1을 곱한다.  


즉, x에서 sigmoid를 거쳐서 y가 나올 때 역전파는 $\frac {\partial L} {\partial y}$에서 $\frac {\partial L} {\partial y} y^2 \exp (-x)$로 흘러간다. 또한 $\frac {\partial L} {\partial y} y^2 \exp (-x)$는 다음처럼 정리가 가능하다.  

$$ \begin{aligned} 
\frac {\partial L} {\partial y} y^2 \exp (-x) &= \frac {\partial L} {\partial y} \frac {1}{(1 + \exp(-x))^2} \exp(-x) \\
&= \frac {\partial L} {\partial y} \frac {1}{1 + \exp(-x)} \frac {\exp(-x)}{1 + \exp(-x)} \\
&= \frac {\partial L} {\partial y} y(1-y)
\end{aligned}$$

위처럼 sigmoid 계층의 역전파는 순전파의 출력(y) 만으로 계산 가능하다. sigmoid 계층은 아래와 같이 구현 가능하다.


```python
class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx
```

## 5.6 Affine/Softmax 계층 구현하기

신경망 순전파에서는 가중치 신호의 총합을 계산하기 때문에 행렬의 내적 (numpy에선 np.dot())을 사용했다.  

> 신경망 순전파 때 수행하는 행렬의 내적은 기하학에선 **어파인 변환 affine transformation**이라고 한다.  



지금까지 계산 그래프는 노드 사이에 scalar 값이 흐른데 반해, 신경망 학습에서는 행렬이 흘러야 한다. 행렬을 사용한 역전파도 행렬의 원소마다 전개해보면 scalar를 사용한 지금까지의 순서로 생각할 수 있다. 실제 전개해보면 다음 식이 나타난다.  

$$ X (1, 2), \, W (2, 3), \, X*W (1, 3), \, B (1, 3), \, Y (1, 3), \, L (scalar)$$



$$ \frac {\partial L} {\partial X} = \frac {\partial L} {\partial Y} * W^T$$

$$ \frac {\partial L} {\partial W} = W^T * \frac {\partial L} {\partial Y}$$



> 이 책에서 벡터는 (1, n)으로 나타내진다.



[벡터, 행렬 미분](https://datascienceschool.net/02%20mathematics/04.04%20%ED%96%89%EB%A0%AC%EC%9D%98%20%EB%AF%B8%EB%B6%84.html#id12) 이곳에서 행렬, 벡터 미분을 공부할 수 있다.  



$X$와 $\frac {\partial L} {\partial X}$는 같은 형상을 가진다. 아래를 보면 알 수 있다.  

$$ X = (x_0, x_1, \cdots , x_n) $$

$$ \frac {\partial z}{\partial X} = (\frac {\partial L} {\partial x_0}, \frac {\partial L} {\partial x_1} , \cdots , \frac {\partial L} {\partial x_n}) $$

행렬 내적 node의 역전파는 행렬의 대응하는 차원의 원소 수가 일치하도록 내적을 조립하여 구한다. 



위에서의 Affine 계층은 입력 데이터로 X 하나만을 고려한 것으로, 데이터 N개를 묶어 순전파하는 배치용 Affine 계층을 생각해본다.  

$$ \frac {\partial L} {\partial X} = \frac {\partial L} {\partial Y} * W^T$$

$$ \frac {\partial L} {\partial W} = W^T * \frac {\partial L} {\partial Y}$$

$$ \frac {\partial L} {\partial B} = \frac {\partial L} {\partial Y}의 \; 열방향 합$$

나머지는 위와 같으나, bias에 주의해야 한다. 순전파 때 bias 덧셈은 X \* W에 대해 각 데이터에 더해진다. 그렇기 때문에 역전파 때는 각 데이터 역전파 값이 편향의 원소에 모여야 한다. (L 자체가 N개의 데이터에 대한 합이기 때문에) 



Affine 구현은 아래와 같다.



```python
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        
        self.x = None
        self.original_x_shape = None
        # 가중치와 편향 매개변수의 미분
        self.dW = None
        self.db = None

    def forward(self, x):
        # 텐서 대응
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        
        self.x = x
        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        dx = dx.reshape(*self.original_x_shape)  # 입력 데이터 모양 변경(텐서 대응)
        return dx
```

마지막 출력층에 사용되는 softmax iwth loss 계층을 알아본다.  

softmax는 입력 값을 정규화하여 출력한다. (출력을 확률, 즉 출력 합이 1로 되도록 정규화한다.)  

> 학습에선 softmax를 사용하지만, 추론 시에는 Affine 계층의 출력 값의 상대적 대소관계만 중요하기 때문에 softmax를 사용하지 않아도 된다. affine 계층의 출력을 **점수 score**라고 한다.



softmax와 loss function인 CEE도 포함하여 softmax-with-loss 계층을 구현해보도록 한다. softmax 계층의 역전파는 $(y_1 - t_1, y_2 - t_2, y_3 - t_3)$를 내놓는다. 역전파에서 '오차'가 앞으로 전해진다.  

> softmax의 loss function으로 CEE를 사용하니 역전파가 $(y_1 - t_1, y_2 - t_2, y_3 - t_3)$로 말끔히 떨어진다. 이 결과는 우연이 아니라, CEE가 그렇게 설계되었기 때문이다. regression의 출력층에서 사용하는 항등함수의 loss function으로 MSE를 이용하는 이유도 같다. 항등함수와 MSE를 사용해도 역전파 결과가 $(y_1 - t_1, y_2 - t_2, y_3 - t_3)$로 말끔히 떨어진다.



학습이 잘 되지 않은 경우, softmax 계층의 역전파의 결과 (오차)가 커서 학습을 크게 진행할 수 있고, 반대의 경우 오차 값이 작아 학습 정도가 작아진다. softmax-with-loss를 구현하면 아래와 같다.



```python
import numpy as np
from codes.common.functions import *

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None # 손실함수
        self.y = None    # softmax의 출력
        self.t = None    # 정답 레이블(원-핫 인코딩 형태)
        
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 정답 레이블이 원-핫 인코딩 형태일 때
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx
```

역전파 시, 전파하는 값을 batch_size로 나눠서 데이터 1개당 오차를 앞으로 전파하는 점을 주의해야 한다.



## 5.7 오차역전파법 구현하기

앞에서 구한 계층을 조합하면 레고를 조립하듯 신경망을 구축할 수 있다.  

신경망 학습의 전체 그림을 정리하면 아래와 같다.  



- 전제  
신경망에는 적응 가능한 weight, bias가 있고, weight와 bias를 training data에 적응하도록 조정하는 과정을 '학습'이라고 한다. 학습은 다음 4단계로 수행한다.  

1. mini-batch  
training data 일부를 무작위로 가져온다. 이를 mini-batch라고 하고, mini-batch의 loss function 값을 줄이는 것을 목표로 한다.
2. 기울기 산출  
mini-batch loss function 값을 줄이기 위해 각 weight의 기울기를 구한다.  
3. parameter 갱신  
parameter를 기울기 반대 방향으로 갱신한다.  
4. 반복  
1~3 단계를 반복한다.

2층 신경망을 TwoLayerNet 클래스로 구현해본다.  
```python
import numpy as np
from codes.common.layers import *
from codes.common.gradient import numerical_gradient
from collections import OrderedDict


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size) 
        self.params['b2'] = np.zeros(output_size)

        # 계층 생성
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()
        
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        
        return x
        
    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
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
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads
```

신경망 layer들을 OrderedDict에 보관한다. 말그대로 순서가 있는 딕셔너리이다. 순전파 때는 추가한 순서대로 forward()를 호출하고, 역전파 때 반대 순서로 backward()를 해주면 된다. 구성요소를 모듈화하면 신경망 구축이 쉬워진다.  



오차역전파법을 사용하면 수치미분보다 빠르지만, 구현이 복잡해 오류가 있을 수 있다. 이때 수치미분법을 사용해 두 방식의 기울기가 거의 같음을 보이는 **기울기 확인 gradient check** 작업을 진행한다.



```python
import numpy as np
from codes.dataset.mnist import load_mnist

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

# 각 가중치의 절대 오차의 평균을 구한다.
for key in grad_numerical.keys():
    diff = np.average( np.abs(grad_backprop[key] - grad_numerical[key]) )
    print(key + ":" + str(diff))
```

<pre>
W1:4.2227636155721954e-10
b1:2.75547187745574e-09
W2:6.030937369643914e-09
b2:1.4057362786035198e-07
</pre>

***

정밀도의 한계 때문에 오차가 0이 되는 일은 드물지만, 0에 가까운 작은 값이 되므로 오차역전파법이 제대로 구현되었다고 볼 수 있다. 다음으로 실제로 오차역전파법을 사용한 신경망 학습을 구현한다.



```python
import numpy as np
from codes.dataset.mnist import load_mnist

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 기울기 계산
    #grad = network.numerical_gradient(x_batch, t_batch) # 수치 미분 방식
    grad = network.gradient(x_batch, t_batch) # 오차역전파법 방식(훨씬 빠르다)
    
    # 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)
```

<pre>
0.10758333333333334 0.1053
0.9060166666666667 0.9084
0.9227333333333333 0.9227
0.9316166666666666 0.9306
0.9436666666666667 0.9426
0.9503166666666667 0.9485
0.9533 0.9502
0.9599666666666666 0.9561
0.9631666666666666 0.9591
0.9647333333333333 0.9614
0.9686833333333333 0.9629
0.9695166666666667 0.9624
0.9735 0.9654
0.9737 0.9662
0.97525 0.9667
0.9769166666666667 0.9678
0.97835 0.9686
</pre>

***

## 5.8 정리

- 계산 과정을 시각적으로 보여주는 계산 그래프 방식이 있다.  
- 신경망의 동작, 오차역전파를 이해하고 layer를 클래스로 모듈화하였다.  
- 모든 layer class에서 forward()와 backward() 메소드를 만들어 순전파, 역전파를 자유롭게 구현 가능하다.  
- layer를 모듈화하면 신경망 layer를 자유롭게 쌓을 수 있다.

