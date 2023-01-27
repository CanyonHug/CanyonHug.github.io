---
title: "밑바닥부터 시작하는 딥러닝 ch.2"
excerpt: "ch.2 퍼셉트론"
date: 2022-11-23
categories:
    - AI
tags:
    - DL
use_math: true
last_modified_at: 2022-11-23
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


# 2. 퍼셉트론 

- 퍼셉트론(perceptron)은 다수의 신호를 받아 하나의 신호를 출력한다. (인공 뉴런으로도 이해된다.)  



## 2.1 퍼셉트론이란?

수용, 연합, 반응 세 부분으로 나눌 수 있는데, 수용층은 입력을 받고 연합층은 받은 입력을 각각 가중치를 곱해 더해지고 반응층에서 특정 임계값 이상이면 1을 출력하고 그보다 낮으면 0을 출력한다. 입력 2개를 받는 퍼셉트론은 아래 수식으로 나타낼 수 있다.   

$$ y = 

\begin{cases}0 \quad (w_1x_1+w_2x_2\leq\theta) \\ 1 \quad (w_1x_1+w_2x_2 \gt\theta) \end{cases} $$



가중치(weight)는 각 신호가 결과에 주는 영향력을 조절하는 요소이다. (가중치가 클수록 해당 신호가 더 중요함을 뜻함.)


## 2.2 단순 논리 회로

입력이 2개, 출력이 1개인 회로를 살펴본다.

- AND 게이트 : 입력이 모두 1일 때만 1 출력, 나머지 0 출력

- NAND 게이트 : (Not AND)를 의미하며 AND 출력을 뒤집은 것과 같다.(둘다 1일때만 0출력)

- OR 게이트 : 입력이 1개 이상 1이면 출력이 1이다.  



퍼셉트론으로 AND, NAND, OR 논리 회로를 표현할 수 있다. $(w_1, w_2, \theta)$의 값을 적절히 조합해 각 논리회로를 표현하면 된다. 퍼셉트론이 논리 회로로 잘 작동하도록 $(w_1, w_2, \theta)$ 값들을 조정하는 것을 **학습**이라고 하고, 학습 시 조정하는 변수들을 매개변수(parameter)라고 한다.

> 사람이 임의로 퍼셉트론의 매개변수값을 조정해가며 논리회로로 작동하도록 할 수 있다. 이 과정을 컴퓨터가 자동으로 하도록 하는 것을 **기계학습(machine learning)** 이라고 칭하는 것이다. 기계학습에서 사람은 퍼셉트론 구조(모델)을 고민하고 컴퓨터에 학습 데이터를 주는 역할을 수행한다.


## 2.3 퍼셉트론 구현하기

AND 함수를 구현하면 다음과 같다.



```python
def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    temp = x1*w1 + x2*w2
    if temp <= theta:
        return 0
    else:
        return 1
    
print(AND(0, 0))
print(AND(0, 1))
print(AND(1, 0))
print(AND(1, 1))
```

<pre>
0
0
0
1
</pre>

***

$\theta$를 -b로 치환하면 퍼셉트론 동작은 다음과 같다.

$$ y = \begin{cases}0 \quad (b + w_1x_1+w_2x_2 \leq 0) \\ 1 \quad (b + w_1x_1+w_2x_2 \gt 0) \end{cases} $$

여기서 b를 **편향(bias)** 라고 한다. 편향은 가중치와 다른 역할을 한다. 가중치는 각 입력 신호가 결과에 주는 영향력(중요도)를 조절하는 매개변수이고, 편향은 뉴런이 얼마나 쉽게 활성화(결과 1을 출력)하는지 조정하는 매개변수이다. (편향으로 인해 입력을 0, 0을 주어도 결과값이 0이 아니다.) bias를 넣어서 함수를 다시 만들면 아래와 같다.



```python
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    temp = np.sum(w * x) + b
    if temp <= 0:
        return 0
    else:
        return 1
```

AND, NAND, OR은 모두 같은 구조의 퍼셉트론이고, 매개변수(weights, bias) 값만 바꿔서 구현 가능하다.



## 2.4 퍼셉트론의 한계

AND, NAND, OR 게이트를 구현한 것처럼 XOR 게이트도 고려해본다. XOR게이트는 exclusive OR로, 입력 2개 중 한쪽이 1일때만 1을 출력한다. 다른 논리 게이트들과 다르게 단일 퍼셉트론은 XOR을 구현하지 못한다. XOR은 (0, 0)과 (1, 1)은 0을 출력하고 (0, 1)과 (1, 0)은 1을 출력해야 한다. 하지만 단일 퍼셉트론은 2차원 평면에 직선을 그어 직선을 기준으로 한쪽을 1, 나머지를 0으로 판단하기 때문에 직선 하나로 XOR을 나타낼 수 없다. 만약 분류기가 비선형적으로 구분한다면 XOR도 구현할 수 있을 것이다. 


## 2.5 다층 퍼셉트론이 충돌한다면

단일 퍼셉트론은 XOR 게이트를 표현할 수 없다. 하지만 퍼셉트론의 진면모는 층을 쌓아 **다층 퍼셉트론 multi-layer perceptron**을 만들 수 있다는 것이다. 퍼셉트론 층을 쌓으면 어떻게 되는지 알아본다.



XOR 만드는 법은 다양하나 기존의 AND, NAND, OR 게이트를 아래와 같이 조합하여 XOR을 만들 수 있다.

```python

def XOR(x1, x2):

    s1 = NAND(x1, x2)

    s2 = OR(x1, x2)

    y = AND(s1, s2)

    return y

```

이는 2층 퍼셉트론과 같다. 0층엔 x1, x2, 1층엔 NAND(x1, x2), OR(x1, x2) 3층에 AND(s1, s2) 값이 저장된다. 

이처럼 단층 퍼셉트론으로 표현하지 못하는 것도 층을 늘리면 구현 가능하다. (직선 한개로 구분되지 못한 것을 직선 2개의 교집합을 이용해 구분할 수 있다.) 

## 2.6 NAND에서 컴퓨터까지
> NAND 게이트만으로 컴퓨터를 만들 수 있다. NAND 게이트는 퍼셉트론으로 만들 수 있다.
> 
> 퍼셉트론으로 층을 쌓으면 비선형적 표현이 가능하고 이론상 컴퓨터가 수행하는 처리도 모두 표현 가능하다.

## 2.7 정리
- 퍼셉트론은 입출력을 갖춘 알고리즘이다.
- 퍼셉트론에서는 weight와 bias를 parameter로 갖는다.
- AND, OR 등의 논리 회로 표현 가능하다.
- XOR은 단층으로 표현 불가능하다.
- 단층은 직선 영역, 다층은 비선형 영역을 표현 가능하다.
- 다층 퍼셉트론은 이론상 컴퓨터를 표현 가능하다.