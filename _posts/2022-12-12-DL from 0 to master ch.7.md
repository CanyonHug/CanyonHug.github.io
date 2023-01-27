---
title: "밑바닥부터 시작하는 딥러닝 ch.7"
excerpt: "ch.7 합성곱 신경망(CNN)"
date: 2022-12-12
categories:
    - AI
tags:
    - DL
use_math: true
last_modified_at: 2022-12-12
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


# 7. 합성곱 신경망 (CNN)

- 이미지 인식, 음성 인식에 사용되는 CNN의 매커니즘, 구현 방식을 알아본다.



## 7.1 전체 구조

**합성곱 신경망 Convolutional Neural Network, CNN**은 지금까지 배운 것처럼 Layer를 쌓아서 구성할 수 있다. 다만 **합성곱 계층 convolutional layer**, **풀링 계층 pooling layer**가 새롭게 등장한다.



지금까지 배운 신경망은 계층끼리 모든 뉴런과 결합되어 있었다. 이를 **완전연결 fully-connected, 전결합**이라고 하고, fully-connected layer를 **Affine 계층**이라는 이름으로 구현했다. 



5층 완전연결 신경망은 다음의 형태로 구성된다.  

![7-1](/assets/images/7-1.png)


Affine - activation function 조합이 4개, 마지막은 Affine - Softmax로 최종 결과(확률)를 출력한다.



반면 CNN의 구조는 다음과 같다.  

![7-2](/assets/images/7-2.png)



'Affine - ReLU' 연결이 'Conv - ReLU - (Pooling)' 으로 바뀌었다. (Pooling 계층은 생략하기도 한다.) 또한, 출력에 가까운 4층은 'Affine - ReLU', 출력층인 5층은 'Affine - Softmax' 조합을 사용한다. 위와 같은 조합을 사용하는 것이 CNN에서 흔히 보이는 구성이다.


## 7.2 합성곱 계층

fully-connected 신경망과의 차이점을 합성곱 계층의 구조를 살펴보며 이해하도록 해본다. fully-connected 신경망에선 Affine 계층을 사용했다. 인접하는 계층의 뉴런이 모두 연결되고 출력의 수는 임의로 결정한다.  



- fully-connected layer의 문제점  

문제점은 '데이터의 형상이 무시된다'는 점이다. 앞에서 MNIST가 데이터를 다룰 때 (1, 28, 28)인 데이터를 (784, )로 변환해 Affine 계층에 입력했다. 이처럼 이미지는 (세로, 가로, 채널(색상) )으로 구성된 3차원 데이터이지만, fully-connected layer에 입력할 때는 1차원 데이터로 변환해 입력하기 때문에 형상이 무시된다. 이렇게 하면, 데이터의 공간적 정보 (픽셀간 거리, RGB 채널 등)를 무시하게 된다.



반면, Conv layer는 이미지를 3차원 데이터로 받고, 3차원 데이터로 전달한다. CNN에서는 Conv layer의 입출력 데이터를 **특징 맵 feature map**이라고 한다. (Conv layer 입력을 **입력 특징 맵 input feature map**, 출력을 **출력 특징 맵 output feature map**이라  한다. 



- 합성곱 연산 Convolution  

Conv layer에선 **합성곱 연산 Convolution**을 처리한다. 이는 이미지 처리에서 말하는 필터 연산에 해당한다.



![7-3](/assets/images/7-3.png)



여기서 데이터의 형상은 (높이 height, 너비 width) = (행 개수, 열 개수)로 표기한다. 위의 예제는 입력이 (4, 4), 필터가 (3, 3), 출력이 (2, 2)인 예제이다.  



Convolution은 **윈도우 Window**를 일정 간격으로 이동하며 **단일 곱셈-누산 Fused Multiply-Add, FMA**를 수행하고 결과를 출력 해당 장소에 저장한다.



- **윈도우 Window** : 입력 데이터와 필터가 겹쳐지는 부분  

- **단일 곱셈-누산 FMA** : 입력과 필터에 대응하는 원소를 곱한 후 총합을 구한다.


Convolution 과정의 순서를 나타내면 다음과 같다. 



![7-4](/assets/images/7-4.png)



완전연결 신경망에서 weight, bias가 존재했던 것처럼 CNN에서는 필터의 매개변수가 weight에 해당한다. bias도 적용하면 다음과 같은 흐름이다. 


![7-5](/assets/images/7-5.png)



bias는 필터를 적용한 후에 더해지고, 항상 하나 (1 * 1)만 존재한다.


- 패딩 Padding  

Convolution을 수행하기 전에 input 데이터 주변을 특정 값 (예컨대 0)으로 채우는 것을 **패딩 padding**이라고 한다. 



![7-6](/assets/images/7-6.png)



위의 예시는 폭이 1인 패딩을 적용한 것으로, input이 (4, 4)에서 (6, 6)으로, output이 (2, 2)에서 (4, 4)로 늘어났다. (padding 폭은 원하는 정수로 설정할 수 있다.)  

> padding은 주로 output 크기를 조정할 목적으로 사용된다. Convolution 연산을 반복해서 거치면 output이 계속 작아서 Convolution을 적용할 수 없게 된다. 이를 막기 위해 padding을 추가해 output 크기를 조절한다.


- 스트라이드 Stride  

필터를 적용하는 위치의 간격을 **스트라이드 stride**(보폭) 라고 한다. stride의 크기에 따라 몇 칸씩 window가 이동할 지 달라진다.



![7-7](/assets/images/7-7.png)



위의 예시에서 stride를 2로 설정한 후 window가 2칸씩 이동한다. stride가 커지면 input을 빨리 커버하기 때문에 output의 크기가 작아진다. 



padding을 키우면 output이 커지고, stride를 키우면 output이 작아진다. 이런 관계를 수식으로 나타내면 다음과 같다.  



( 입력 크기 ($H, W$), 필터 크기 ($FH, FW$), 출력 크기 ($OH, OW$), 패딩 ($P$), 스트라이드 ($S$) )  

$$ OH = \frac {H + 2P - FH} {S} + 1 $$



$$ OW = \frac {W + 2P - FW} {S} + 1 $$



식을 떠올릴 때, padding이 더해진 $H + 2P$에서 filter의 끝부분 $FH$이 $S$만큼 이동하며 끝까지 가는 경우를 생각하면 식이 자연스레 도출될 것이다.



이때 $\frac {H + 2P - FH} {S}$과 $\frac {W + 2P - FW} {S}$이 정수로 나눠 떨어져야 한다. 딥러닝 프레임워크 중엔 정수가 아닐 때 반올림하는 등 정수가 아닐 때 처리를 해주기도 한다.


- 3차원 데이터의 합성곱 연산  

지금까진 채널이 1인 2차원 형상에 대해 다뤘지만, 이미지는 보통 (행, 열, 채널 (RGB = 3) )로 구성된 3차원 데이터이다. 3차원 데이터 Convolution 하는 방법도 2차원과 크게 다르지 않다.



![7-8](/assets/images/7-8.png)



3차원 Convolution은 채널이 커졌기 때문에 input, filter가 늘어난다. input data와 filter의 Convolution을 채널마다 수행하고 이를 더해서 하나의 출력을 얻으면 된다.


3차원 데이터 Convolution 계산 순서는 다음과 같다.  



![7-9](/assets/images/7-9.png)



3차원 데이터 Convolution을 진행할 때 input data 채널 수와 Filter 채널 수가 같아야 한다. (모든 채널의 필터는 같은 (행, 열)을 가져야 한다.)


- 블록으로 생각하기  

3차원 합성곱 연산은 데이터와 필터를 직육면체 블록이라고 생각하면 쉽다. 여기선 데이터를 (채널 C, 높이 H, 너비 W) 순서로 나타낸다.



![7-10](/assets/images/7-10.png)



필터를 1개 사용하면 output feature map의 채널이 1개가 된다. 필터를 여러 개 사용하면 output의 채널도 커진다.


![7-11](/assets/images/7-11.png)



위와 같이 필터를 FN개 적용하면, output feature map도 FN개가 생성된다.  

따라서 필터의 가중치는 (출력 채널 수, 입력 채널 수, 높이, 너비)로 4차원 데이터이다. bias는 채널 하나에 값 하나씩 대응되어 더해지기 때문에 (FN, 1, 1)의 형상을 가진다. 


- 배치 처리   

완전 연결 신경망과 마찬가지로 CNN에서도 데이터를 한 덩이로 묶어 배치로 처리한다. 각 계층을 흐르는 데이터는 한 차원 높여 (데이터 수, 채널 수, 높이, 너비)로 4차원 데이터로 구성된다. 



![7-12](/assets/images/7-12.png)



위에서처럼 신경망에 4차원 데이터가 흐를 때마다 N개의 데이터의 Convolution 연산을 처리하게 된다.


## 7.3 풀링 계층  

padding이 데이터를 세로, 가로 방향으로 넓혔다면, **풀링 Pooling**은 세로, 가로 방향의 공간을 줄이는 연산이다. pooling은 대상 영역의 평균을 취하는 **평균 풀링 average pooling**, 영역 내 최댓값을 취하는 **최대 풀링 max pooling** 방식이 있다.



![7-13](/assets/images/7-13.png)



위의 예시는 2 \* 2 크기의 window, stride 2의 값으로 max pooling을 수행하는 과정을 나타낸다. (보통 pooling의 윈도우 변 크기와 stride는 같은 값으로 설정한다.) 이미지 인식 분야에서는 주로 max pooling을 사용하고 이 책에서 말하는 pooling layer는 max pooling을 말할 것이다.



Pooling layer의 특징으로 다음의 것들이 있다.  

- 학습해야 할 parameter가 없다.  

- 채널 수가 변하지 않는다. (채널별로 독립적으로 pooling 연산 진행)  

- 입력 data의 변화에 영향을 적게 받는다.(강건하다. robust 하다 표현함.)


## 7.4 합성곱 / 풀링 계층 구현하기  

이전에 layer들을 class로 구현했던 것처럼 Conv, Pooling layer를 구현해본다.  



우선 4차원 데이터를 알아본다. (10, 1, 28, 28)은 10개의 1채널, 28 높이, 28 너비의 데이터이다.



```python
import numpy as np
x = np.random.rand(10, 1, 28, 28)
print(x.shape)

print(x[0].shape)
print(x[1].shape)

print(x[0, 0].shape)
print(x[0][0].shape)
```

<pre>
(10, 1, 28, 28)
(1, 28, 28)
(1, 28, 28)
(28, 28)
(28, 28)
</pre>

***

- im2col로 데이터 전개하기  

Convolution 연산을 그대로 구현하려면 for 문을 겹겹이 써야한다. 하지만 numpy에선 for 문을 사용하면 성능이 떨어지기에 원소에 접근할 때 for 문을 사용하지 않는 것이 좋다. 따라서 for 문 대신 **im2col, image to column**이라는 함수를 통해 구현해본다. im2col은 입력 데이터를 2차원 행렬로 바꿔준다.



![7-14](/assets/images/7-14.png)



필터의 적용 영역이 겹치는 경우, im2col로 전개한 후의 원소 수가 원래 블록의 원소 수보다 많아진다. 하지만, 행렬 계산 라이브러리가 최적화되어 있기 때문에 행렬 계산으로 문제를 변환하면 라이브러리를 활용해 효율을 높일 수 있다.


im2col로 입력 데이터를 전개한 후엔 filter를 세로로 1열로 전개한 뒤, 두 행렬의 내적을 계산하면 된다.  



![7-15](/assets/images/7-15.png)



행렬 계산을 통해 나온 데이터는 2차원이므로 다시 4차원 데이터로 reshape 해주면 된다.



im2col 구조와 사용 예시는 아래와 같다.



```python
def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    다수의 이미지를 입력받아 2차원 배열로 변환 (평탄화)
    
    Parameters
    ----------
    input_data : 4차원 배열 형태의 입력 데이터(이미지 수, 채널 수, 높이, 너비)
    filter_h : 필터의 높이
    filter_w : 필터의 너비
    stride : 스트라이드
    pad : 패딩
    
    Returns
    -------
    col : 2차원 배열
    """

    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col
```


```python
x1 = np.random.rand(1, 3, 7, 7) # 데이터 수, 채널 수, 높이, 너비
col1 = im2col(x1, 5, 5, stride=1, pad=0)
print(col1.shape)    # (9, 75)
 
x2 = np.random.rand(10, 3, 7, 7) # 데이터 10개
col2 = im2col(x2, 5, 5, stride=1, pad=0)
print(col2.shape)    # (90, 75)
```

<pre>
(9, 75)
(90, 75)
</pre>

***

높이 7, 너비 7인 데이터를 5 \* 5, 채널 3인 필터가 stride 1로 움직이기 때문에, 3 \*3 = 9, 5 \* 5 \* 3 = 75가 된다.



im2col을 활용해 Conv 계층을 구현하면 아래와 같다.



```python
class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W  # 필터 (개수, 채널, 높이, 너비)
        self.b = b  # 편향 (개수, )
        self.stride = stride
        self.pad = pad
    
    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2 * self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2 * self.pad - FW) / self.stride)

        # col.shape = (N * out_h * out_w, C * FH * FW)
        col = im2col(x, FH, FW, self.stride, self.pad)

        # col_W.shape = (C * FH * FW, FN)
        col_W = self.W.reshape(FN, -1).T

        # np.dot(col, col_W).shape = (N * out_h * out_w, FN)
        # self.b.shape = (FN, )
        # broadcasting
        # out.shape = (N * out_h * out_w, FN)
        out = np.dot(col, col_W) + self.b

        # out.shape = (N, FN, out_h, out_w)
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        
        self.x = x
        self.col = col
        self.col_W = col_W
        
        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        # dout.shape = (N, FN, out_h, out_w)
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)
        # dout.shape = (N * out_h * out_w, FN)
        self.db = np.sum(dout, axis=0)
        
        # dW = col.T * dout
        # col.T.shape = (C * FH * FW, N * out_h * out_w)
        self.dW = np.dot(self.col.T, dout)
        # dW.shape = (C * FH * FW, FN)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)
        # dW.shape = (FN, C, FH, FW)

        # dcol = dout * col_w.T
        # dout.shape = (N * out_h * out_w, FN)
        # col_w.T.shape = (FN, C * FH * FW)
        dcol = np.dot(dout, self.col_W.T)
        # dcol.shape = (N * out_h * out_w, C * FH * FW)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx
```

col에 입력데이터를 im2col을 활용해 전개하고, col_w에 필터도 reshape을 사용해 2차원 배열로 전개한다. reshape(FN, -1)의 두번째 인수에 -1은 다차원 배열의 원소수를 변환 후에도 똑같이 유지되도록 적절히 묶어주는 기능을 수행한다. ([참고](https://domybestinlife.tistory.com/149))



또한 reshape(FN, 01).T 에서 T를 통해 행렬을 전치하여 필터 1개가 1개의 열로 세워지에 만든다.



![7-16](/assets/images/7-16.png)



dot계산과 bias를 더한 후의 out.shape이 (N \* out_h \* out_w, FN)이므로, reshape(N, out_h, out_w, -1)로 분리한 후에 transpose로 FN을 2번째 자리로 끌고온다.  

> (tranpose()는 T보다 확장된 method로 배열을 원하는 방향으로 바꿀 수 있다. [참고](https://pybasall.tistory.com/124))



Conv layer의 역전파를 구현할 때는 dout을 행렬 형태로 변환하고, 순전파 과정을 역으로 처리한다. (차원 수를 맞춰 Affine layer 역전파와 동일하게 처리한다.) dx는 dcol을 im2col을 역으로 구현한 col2im 함수를 사용해 구해주면 된다. col2im 구현은 다음과 같다.



```python
def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """(im2col과 반대) 2차원 배열을 입력받아 다수의 이미지 묶음으로 변환한다.
    
    Parameters
    ----------
    col : 2차원 배열(입력 데이터)
    input_shape : 원래 이미지 데이터의 형상（예：(10, 1, 28, 28)）
    filter_h : 필터의 높이
    filter_w : 필터의 너비
    stride : 스트라이드
    pad : 패딩
    
    Returns
    -------
    img : 변환된 이미지들
    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    # dcol.shape = (N * out_h * out_w, C * FH * FW)
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)
    # col.shape = (N, C, FH, FW, out_h, out_w)
    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]
```

- 풀링 계층 구현하기  

pooling layer 구현도 Conv layer와 마찬가지로 im2col을 사용해 입력 데이터를 전개한다. 하지만, pooling은 채널이 독립적이란 점이 Conv layer와 다르다. (pooling 적용 영역을 채널마다 독립적으로 전개한다.)



![7-17](/assets/images/7-17.png)



위와 같이 전개한 후, 행렬에서 행별 최댓값을 구하고 reshape 과정을 거치면 된다.


![7-18](/assets/images/7-18.png)



위의 과정이 pooling layer의 forward 처리 흐름이다. 



```python
class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        
        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        # col.shape = (N * out_h * out_w, C * pool_h * pool_w)
        col = col.reshape(-1, self.pool_h*self.pool_w)
        # col.shape = (N * out_h * out_w * C, pool_h * pool_w)
        

        # np.argmax() : return max value index in ndarray
        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        # out.shape = (N * out_h * out_w * C, )
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        # out.shape = (N, C, out_h, out_w)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        # dout.shape = (N, out_h, out_w, C)
        
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        # dmax.shape = (N * out_h * out_w * C, pool_h * pool_w)
        # ndarray[a_array, b_array] : (a_array 원소, b_array 원소) index에 접근
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        # dmax.shape = (N, out_h, out_w, C, pool_h * pool_w)
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        # dcol.shape = (N * out_h * out_w, C * pool_h * pool_w)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx
```

max gate를 미분할 때는 max로 뽑힌 변수만 1이고 나머지는 0이기 때문에, max로 뽑힌 변수에 대해서만 dout값을 가져오면 된다. 이를 위해 forward 단계에서 arg_max를 기록해 둔다. dmax를 dcol 형태로 변환 후, col2im 함수를 통해 다시 image 형태로 변환해주면 backward 과정이 마무리된다. forward에서 흘러가는 차원 변환 흐름을 거꾸로 따라가면 된다. 중간에 행렬 col, col_w 끼리 미분은 Affine layer와 동일하다.


## 7.5 CNN 구현하기

Conv layer와 Pooling layer를 구현했으니, layer들을 조립해 CNN을 구현해본다.



![7-19](/assets/images/7-19.png)



위의 순서대로 CNN을 구성하고 이를 SimpleConvNet 이라는 클래스로 구현해본다.  

(Conv layer의 hyper-parameter는 dict 형태로 주어진다.)



```python
import pickle
import numpy as np
from collections import OrderedDict
from codes.common.layers import *
from codes.common.gradient import numerical_gradient


class SimpleConvNet:
    """단순한 합성곱 신경망
    conv - relu - pool - affine - relu - affine - softmax
    Parameters
    ----------
    input_size : 입력 크기（MNIST의 경우엔 784）
    hidden_size_list : 각 은닉층의 뉴런 수를 담은 리스트（e.g. [100, 100, 100]）
    output_size : 출력 크기（MNIST의 경우엔 10）
    activation : 활성화 함수 - 'relu' 혹은 'sigmoid'
    weight_init_std : 가중치의 표준편차 지정（e.g. 0.01）
        'relu'나 'he'로 지정하면 'He 초깃값'으로 설정
        'sigmoid'나 'xavier'로 지정하면 'Xavier 초깃값'으로 설정
    """
    def __init__(self, input_dim=(1, 28, 28), 
                 conv_param={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1},
                 hidden_size=100, output_size=10, weight_init_std=0.01):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))

        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * \
                            np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['W2'] = weight_init_std * \
                            np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * \
                            np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        # 계층 생성
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                                           conv_param['stride'], conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        acc = 0.0
        
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt) 
        
        return acc / x.shape[0]

    def numerical_gradient(self, x, t):
        loss_w = lambda w: self.loss(x, t)

        grads = {}
        for idx in (1, 2, 3):
            grads['W' + str(idx)] = numerical_gradient(loss_w, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_w, self.params['b' + str(idx)])

        return grads

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W3'], grads['b3'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads
        
    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, key in enumerate(['Conv1', 'Affine1', 'Affine2']):
            self.layers[key].W = self.params['W' + str(i+1)]
            self.layers[key].b = self.params['b' + str(i+1)]
```

위의 SimpleConvNet을 MNIST 데이터셋으로 학습시키면 아래와 같다. (training data에 대한 정확도는 99.92%, test data에 대한 정확도는 98.96%가 된다.)



```python
import numpy as np
import matplotlib.pyplot as plt
from codes.dataset.mnist import load_mnist
from codes.common.trainer import Trainer

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

# 시간이 오래 걸릴 경우 데이터를 줄인다.
x_train, t_train = x_train[:5000], t_train[:5000]
x_test, t_test = x_test[:1000], t_test[:1000]

max_epochs = 20

network = SimpleConvNet(input_dim=(1,28,28), 
                        conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)
                        
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=max_epochs, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()

# 매개변수 보존
network.save_params("params.pkl")
print("Saved Network Parameters!")

# 그래프 그리기
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
```

<pre>
Saved Network Parameters!
</pre>
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAYIAAAEKCAYAAAAfGVI8AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAgAElEQVR4nO3de3xcdZ3/8ddnJpN7mrTpPW1paUuhILYQ8MJlQcRSvFB2vYCIipeCgrffjy7w21VxXXdx+elP+a3AoqICKnKH1SKIou7+EEsLhdIbTaGXNGmapE2aezIz398f56RMJzPJNM3JpJn38/GYx5zrnM+cTL6fc77ne77HnHOIiEjuCmU7ABERyS4lAhGRHKdEICKS45QIRERynBKBiEiOUyIQEclxgSUCM7vbzPaZ2atp5puZ3WZmNWb2ipmdFlQsIiKSXpBnBD8FLhpk/nJgof9aCdwRYCwiIpJGYInAOfdnYP8gi1wC3OM8zwMVZjYjqHhERCS1vCxuuwrYnTBe60+rT17QzFbinTVQUlJy+oknnjgqAYrI+NDS2cfeg930xeJEwiGmTyikojgyatve09JFPKEXh5AZVRVFVBRHcA4cjrgD55w37iBO/7A7NJ4fDlMYGd7x+7p165qcc1NSzctmIrAU01L2d+Gcuwu4C6C6utqtXbs2yLhEZIQ99tIebn1qK3UtXcysKGLVskWsWFo1atu+6ZENTO6LHZoWzgvx2QtP4KwFk+nsjdHVF6OrN0ZXX5Su3jidvVG6+7zpnb0xb7g3Rl/cEYs5Ys4Ri6d4OUc07ojH33xvaWxnWnxg0RYFmo7wu3zqb+Zz4/LhHQib2c5087KZCGqB2Qnjs4C6LMUiMq5lsyB+9MVabnp0A919cQD2tHTx9w+9wou7DnDSjAm0d0dp64nS3h2lvaeP9p4o7T0x2rv9YX9+R0+UFOXpsPRE4/zrk1uGXC4vZBTlhymKhCnKDxMJhwibEQ55r1DIyAvZoWn5odCheWF/+taGtrSf/8ULFlKQFyI/HCI/z38lDueFKEgYnzahcGR2QPL3DORTM/MEcJ2Z3Q+8DWh1zg2oFhIZD8bCEXGXf0S8p6WLmx7ZAJBxDLG4oycao607SmtXHy2dff57L61dfRzs6qOlq++wef2v/R29Az6vNxbnnr8cfoBaWpDnvQq997LCPKaWFR4aLy3IIxxKVZEwuO//flvaeT/8ePWhQr4oEqY43xsu9Icj4aO/jHrWLX9gT0vXgOlVFUX8jwtPOOrPHwmBJQIz+yVwHjDZzGqBrwMRAOfcncBq4GKgBugErgoqFpFsSlUQ3/jIK3T2RbnwpOn0xuL0RhNesRg9h42/ORyNO+LOEY35736VxKGqiKSqiVjc8eDa3Ye23a+rL8YND7/Cr17Yffj2YwO32RuLExviUNwMJhRGqCiOUF7kvWZNLKK8KMLP/7or9TrAcze9i9KCPEry8wgNo5DPxEPratMWxBcunhbINhOtWrbosL8/QFEkzKpliwLfdqYCSwTOucuHmO+Aa4Pavshocs6xv6OX+tZuGg52U9/azd5W7/3Xr9TRE40ftnx3X5z/9cir/C9S3mYzbCHDq7Iwr8oiFDI6emMpl+2JxonG4xRGQkwozPOrH8KHqiYKUlRVlBXmUV4UoaIo/1CBX14coawgfUH+x62NKQvimRVFzCgvGtHvn0q2C+L+s65snRFmIptVQyLHBOccBzr72LW/k72tXYcV8ntbu9l70Hv1JhX24ZAxtaxgQBJI9M0VpxyqA46kqCdOLozzQgPrp/sL/XDIMBtYGA9WNfHgNe88+h00BBXEXgxjqeBPpkQgOWGoOvpoLE59aze79neys7mTnfs72NXsDe/e30lbT/Swz8sPh5heXsj08kKWzqlg+gRveEZ5IdPLi5hRXsjk0gLCIRu0IL7y7ccF/t2zXhA/cx4rwvsgnDTjmamwNH39/YjGkM2C+NaF0LFv4PSSqbBqdL7/UJQIZNxLVUd//YMv8+Da3YTDIXY1d1B7oItoQj14JGzMnljMnMpiqudOZM6kYuZMKvarMwqZVJKf8ug7lWO2IHYOejuguwW6WqC7FfIKIL8UCsq8V34phIa4oJqqEBxs+kjLdkE8Et8/HodYD1jI+xuMMCUCGXdaO/uoaWxjW0M7Nfvauff5nQOqZ6Jxx3PbmzmlqpyTZ5az/C0zOG6SV/DPmVTMjPKiYbVQSSXrVRODFUTP/gt0HfAL+pak4RaI9w39+fllbyaGw14TvPfBbHocLAyhPAiFvYKufziU588LvzkeikBePoQLvAIxnO+9h/K8K9ZH+v37xaLQ1wl9XdDX4b33dvrTOt+cF+uDeBRc3HuPx/zxmD+cYnwwP3s/RHu9Qr7/PdY7cFrcPyM9+yvw7psH/8xhsGPtmcW6oezYNNLNJ51zNHf0UrOvnW372qlpaPPe97Wzr63n0HIFeaG0dfQGvHHLe4cdQ8ZG44i0qwVad0PLbmjZ5Q/v8l716wdZ0aBwAhRWQNFEKPLfE8f7hwsneAVhz0HoaUt6pZrWBj3t0NM6Mt9xUOYnhoKEROG/N25Ov1phuV/AD2ziOmyhvIQkljf495/99tSJbcC7/31mnQlzzxpWWGa2zjlXnWqezggkcJm2Y4/HHZ19sYQbi94c7m+/vr2xg+372tm2r40DnW8erZYW5LFgainnnjCFhVNLWTC1lIVTy6iaWETLP82lkpYBcTVTAaS92dIryJproHm7935gh/fPnV8MkSKIlHjv+cUQSXwlTRvsiLTrgHfa72JJR5gpjjhjfXCwLkWBv3tgYZNXCOWzoWLO4H+crzV7R9tBurk8/bzPPZfwPWOH74fko+x41NsHsR6I9h85J773H0H3Hj5tsERw6mX+38v/Wx72N0wxLZyfcLaS6uwlRTXZYN//009lvh8DpEQggehvabPnQBf/9OuNKduxr3roZX7wbM2hu0fbe6MMdYJaURzhhKllXHTKDBZOLWXhNK/Qnz6hMG2dfaokcGh6tBdadnoFfdO2wwv+9r0JSxuU+X0i9lcVjMRR5LfnDm+9gglvFvTHvdMf9sfL50DJ5DerSgYriIJOAkOZdnLw2xjs+1/8b8Fv/xigRCDDEo87Gtt7qD3QxZ6WLvYc6KL2QOeh4T0tXXSmab/ery/mWDit1L9rNEJpYR5l/p2lJQVvDpcW5DEh1kJ5xw5KInGMXryObf3ObZv913B8a7p35NmvuBIqF8CCd0PlfG+4cgFMmucdESaKRSGaXJfc5b33Jow//vn021/2r/5RZejw6oT+evHDxvOgbLpX6BdVDPMLZ0HJ1PRVY7ngGPj+SgQyqGgsztaGNtbvbuHVPa3s2t/JngNd1LV00xs7vO69ojhCVUUR8yaXcM7CKVRNLKKqoojqh97G5DRVM5VXJFXN9HZC4xZo2Ag7Nnnv+zZBR2MwX/Dsr8DkhX5hfzwUT8p83XAehMuGviA6WCJ4xyDzRkq2C6JsN5HM9e+fASUCOUzDwW5e2tXCS7sPsH5XC6/Uth6q1plYHOG4yhJOripn2SnTmVVR5Bf2xVRNLKK0IM3P6aFBqmY2PQ4Nm2DfRu99/+sc6oQ2rwimnggLl3lVCFNO8KYdqZ9enH7eBV898s871hwDBVGgcv37Z0CJYLwbpMVK95e38OqeVl7a1cL63S28tOsAda3dgNeOfvHMcj5yxmyWzqlg6eyJzJ5UlHHbeaI90NEEnUN0tPvAxwHzjsanLYa3fMgr9KedDBPnZr8OeyRk+4hUZAhKBOPdIC1WTvn6U4duopo9qYjT507iM7MrWDKngsUzJlAY8QvhaC/0tsOBBuhs9l79hXxHU8J4sz+tGXrTd717mM8+C1NO9FrZBCXbBbGOSGWMUyIYh5rbe9i6t40te9v41CDL3T3/z8wuiTK9oI+ieKfXXHJbG2xMagse7U7/IeF8KJ4MJZXe+6R5h48XV8IDV6Zfv+q0YX/PjKkgFhmUEsExrLsvxraGdrbsPcjWvW1sbWhjc30bB9o7OdF2syRU43f8ndq5u+/w7tQsnOB3G+DfCVo6HSoXDrxDtKDMK9iLK98s6AvK0t/RKSLHBCWCY0Rbdx9/2d7M5vo2tjYcZMveNnY0dRB3MI39nBF5nRUlO/lG/jZmF79GJD7IUXy/f9wXSL8lA2S7akZEBqVEMIa1dvbxu80NPLmhnv/a1kRvLE6R9fDu8nquLdnJW2bWMLtzI0Vd/o1Pffkw/VSYdRXMqvZe339r+g2MRhIAVc2IjHFKBGNMc3sPT29qYPWGev6yvZlo3DG7PJ9/O2Ez7z74KCX7N2LdMejGa1Uz/2yYdYZX6E9/y8DCXUfjIjIEJYIxoOFgN09t3MuTG/by1zeaiTs4rrKYz5w9j4+Ub2Tuy/+MvbEJpp3i3QA1qxqqqqF0ytAfrqNxERmCEsEoSNXz5hnzJvHkhnp+++pe1u06gHOwYGop156/gOWnzOCk3lewZ74Ia9bApPnwwZ/A4hVD9/0uInKElAgClqrnza88sP5Q52onzZjAV959AstPmc7CaWVQ/zL8/iqoecbr5Oz934clV0B4kOY/IiJHQYkgYLc+tXVAz5vOwYTCPJ647mzmTi7xJjZvhwe/CBsf8fp+v/CbcOZnB3Z0JiIywpQIAlaX4lm1AG3dUS8JHKyDP30bXrzXu9B77ip45xe8B2aIiIwCJYIARWNx1hR8jik28AlF+ymH330S/vof3oM3zvgMnHs9lKo1j4iMLiWCgHT3xfjCL1/ihymSAMAkWuH/3QZvvQzOu9FrCioikgVKBAE42N3HZ3+2ljU79sNg92x97jmvx00RkSxSW8QR1tTew+V3Pc+6nQf43keWDL6wkoCIjAFKBCOo9kAnH7rzL2xvbOeHn6jmkiVVQ68kIpJlqhoaIdsa2rjyx2vo7I1y36ffRvXcI3jkoYhIFikRjICXdh3gqp++QCQc4ldXv4OTZkzwZqz5YfqV1NePiIwRSgRH6b+2NXL1veuYXFrAfZ9+G3Mq/SdtvfIArL4eFl0MH75HdwaLyJilRHAUVm+o50v3v8T8KaXc86kzmTqh0Jux9Ul49BqYe47XR5CSgIiMYUoEw/SLv+7iHx7bwOlzJvLjT55BeZFf2O/4b3jwkzDjVLj8lxApzGqcIiJDUSI4Qs45bv/jdm59aivnL5rC7VecTlG+/5D3PS/CLy7zbg674mHvMY4iImOcEsERcM7xL6s388P/eoMVS2Zy64feSiTst8Bt3Ar3/Z3XYdyVj3rP9BUROQYoEWQoFnfc8PArPLSulk++cy5fe99iQiH/oe0tu+DeSyGUBx9/DCbMzG6wIiJHINAbyszsIjPbamY1ZnZjivnlZvafZvaymW00s6uCjOdo/GHLPh5aV8sX3rWAr78/IQm074N7LoHedu9MoHJ+dgMVETlCgSUCMwsDPwCWA4uBy80suU+Fa4FNzrm3AucB3zGz/KBiOhrb9rUBcPXfzMfMTwJdLXDv30LbXrjiIZh+ShYjFBEZniDPCM4EapxzrzvneoH7gUuSlnFAmXklaymwH4gGGNOw7WzqZHJpAaUFfm1abyf84iPQuAU+ch/MPjO7AYqIDFOQiaAK2J0wXutPS/TvwElAHbAB+JJzLp78QWa20szWmtnaxsbGoOId1M79HRzXf7NYtBceuBJq18Df/QgWXJCVmERERkKQicBSTHNJ48uA9cBMYAnw72Y2YcBKzt3lnKt2zlVPmTJl5CPNwM7mTi8RxGPw6ErvmcLv/z6cvCIr8YiIjJQgE0EtMDthfBbekX+iq4BHnKcGeAM4McCYhqW7L0Z9azdzJxXDr78CGx+F9/wznPbxbIcmInLUgkwELwALzWyefwH4MuCJpGV2ARcAmNk0YBHweoAxDcuu/Z0ALG+4E178GZxzvfdcYRGRcSCw+wicc1Ezuw54CggDdzvnNprZNf78O4FvAj81sw14VUk3OOeagoppuHY0dfCO0EYWbvsxVH8a3vWP2Q5JRGTEBHpDmXNuNbA6adqdCcN1wHuCjGEk7Gzu5FTzT1Qu+BpYqssfIiLHJj2hLAM7mjs4MbLXe4ZAUUW2wxERGVFKBBnYtb+TE/P2wuQTsh2KiMiIUyLIwI6mdubEa2HygmyHIiIy4pQIhtAbjdPVso+SeJvOCERkXFIiGELtgU7m9t/+oEQgIuOQEsEQdjZ3Mj9U741MXpjdYEREAqBEMIQdzR3MtzpcuADKZw+9gojIMUaJYAg7mzs5IVzvPWcgFM52OCIiI06JYAg7mztYGK7HdH1ARMYpJYIh1DW1MiPeoOsDIjJuKREMIhqLE2p5gxBxtRgSkXFLiWAQ9a3dzHF7vBGdEYjIOKVEMIj+FkMAVCoRiMj4pEQwiB3NncwP1RErnQEFpdkOR0QkEEoEg9jZ1MGCUD2hKbo+ICLjlxLBIHY0dbDA1HRURMY3JYJBtDfXUkKnLhSLyLimRJBGPO6ItGz3RpQIRGQcUyJIo6Gtmznx/qajqhoSkfFLiSCNHU2dzLc6YnnFUDYz2+GIiARGiSCNnc0dHG/1xCYeDyHtJhEZv1TCpdF/D0HetBOzHYqISKCUCNKoa2ymypoI6fqAiIxzSgRpxBprCOH0wHoRGfeUCFJwzlHQ+ro3ojMCERnnlAhSaGrvZVasFofBpPnZDkdEJFBKBCnsbO5gfqiO7pKZkF+c7XBERAKlRJDCjuZOjrc6nLqeFpEcoESQws6mNuZbPQXT1XRURMa/vGwHMBa1NOyi2HpA3U+LSA7QGUEK1vSaN6DO5kQkBygRpFB0UE1HRSR3KBEkaensZWZ0N73hUiidlu1wREQCp0SQZEez1+toV/k8MMt2OCIigQs0EZjZRWa21cxqzOzGNMucZ2brzWyjmf0pyHgysbO5g+NDejyliOSOwFoNmVkY+AFwIVALvGBmTzjnNiUsUwHcDlzknNtlZlODiidTe/Y2contp2/mSdkORURkVAR5RnAmUOOce9051wvcD1yStMxHgUecc7sAnHP7AownI917twIQmbooy5GIiIyOIBNBFbA7YbzWn5boBGCimf3RzNaZ2cdTfZCZrTSztWa2trGxMaBw/W3tr/EG1HRURHJEkIkg1ZVWlzSeB5wOvBdYBnzVzAZUzjvn7nLOVTvnqqdMmTLykSYobXudOCGYdHyg2xERGSsySgRm9rCZvdfMjiRx1AKzE8ZnAXUplvmtc67DOdcE/Bl46xFsY0S1dfcxI7qbg0VVkFeQrTBEREZVpgX7HXj1+dvM7BYzy6QTnheAhWY2z8zygcuAJ5KWeRw4x8zyzKwYeBuwOcOYRtzO5k7mWz295TobEJHckVEicM4945y7AjgN2AH8zsyeM7OrzCySZp0ocB3wFF7h/oBzbqOZXWNm1/jLbAZ+C7wCrAF+5Jx79Wi/1HDtbGpjntUT1oViEckhGTcfNbNK4GPAlcBLwM+Bs4FPAOelWsc5txpYnTTtzqTxW4FbjyTooDTv2U6h9UHV4myHIiIyajJKBGb2CHAicC/wfudcvT/rV2a2NqjgRltfwxYACtX9tIjkkEzPCP7dOfeHVDOcc9UjGE9W5R3Y7g2o6aiI5JBMLxaf5N8FDICZTTSzzwcUU9aUtb9BR3gCFFdmOxQRkVGTaSL4rHOupX/EOXcA+GwwIWVHV2+MmdHdtJbMVWdzIpJTMk0EIbM3S0e/H6H8YELKjl37Ozne6olWLMh2KCIioyrTRPAU8ICZXWBm7wJ+idfsc9yora9nqrUQmaamoyKSWzK9WHwDcDXwObyuI54GfhRUUNlwsNbrFHXCLDUdFZHcklEicM7F8e4uviPYcLInus97TnGJ7iEQkRyT6X0EC4F/BRYDhf3TnXPjpi+GgpbtRAmTN/G4bIciIjKqMr1G8BO8s4EocD5wD97NZeNGeccbNOVXQThljxkiIuNWpomgyDn3e8CcczudczcD7wourNHVE40xM1pLe+m4OcEREclYpomg2++CepuZXWdmlwJZf6zkSKltbuM420ts0vxshyIiMuoyTQRfBoqBL+I9SOZjeJ3NjQv7dm0l32IUqI8hEclBQ14s9m8e+7BzbhXQDlwVeFSjrL3WewRCxZyTsxyJiMjoG/KMwDkXA05PvLN4vIk3ek1Hy3UPgYjkoExvKHsJeNzMHgQ6+ic65x4JJKpRVnhwOwesgonFE7MdiojIqMs0EUwCmjm8pZADxkUimNS5k6bCOSgNiEguyvTO4nF3XaBfNBanKlbLrrILsh2KiEhWZHpn8U/wzgAO45z71IhHNMr21u9hlrWxq1IPoxGR3JRp1dCvE4YLgUuBupEPZ/Q17dzILKBwhpqOikhuyrRq6OHEcTP7JfBMIBGNsq46r9fRyuNOyXIkIiLZkekNZckWAnNGMpBscU019LgIlVV6II2I5KZMrxG0cfg1gr14zyg45hW3vU5deCbz8jKtJRMRGV8yrRoqCzqQbJncvZOGogXMy3YgIiJZklHVkJldamblCeMVZrYiuLBGR7yvh+mxvXSVq7M5EcldmV4j+LpzrrV/xDnXAnw9mJBGT9PuLeRZHJuspqMikrsyTQSpljvmK9X373wVgJKZJ2U5EhGR7Mk0Eaw1s++a2XwzO97M/g+wLsjARkNP/VYAJs9V01ERyV2ZJoIvAL3Ar4AHgC7g2qCCGi22fxt73URmTJ2S7VBERLIm01ZDHcCNAccy6sra3qAubzbTQ+O2h20RkSFl2mrod2ZWkTA+0cyeCi6sUeAcU3p30VI8N9uRiIhkVaZVQ5P9lkIAOOcOcIw/s9i176PUddBToaajIpLbMk0EcTM71KWEmc0lRW+kx5KW3RsByJtyQpYjERHJrkybgP4D8N9m9id//FxgZTAhjY7W3ZuYCJTo8ZQikuMyvVj8WzOrxiv81wOP47UcOmb17t1KpytgxmxVDYlIbsv0YvFngN8D/9N/3QvcnMF6F5nZVjOrMbO0rY7M7Awzi5nZBzML++jlHajhDTedqoklo7VJEZExKdNrBF8CzgB2OufOB5YCjYOtYGZh4AfAcmAxcLmZDaiH8Zf7NjCqrZAmdLxBfWQO+XnD7YlbRGR8yLQU7HbOdQOYWYFzbguwaIh1zgRqnHOvO+d6gfuBS1Is9wXgYWBfhrEcvb4uJvXt5WDJ3FHbpIjIWJVpIqj17yN4DPidmT3O0I+qrAJ2J36GP+0QM6vCe+zlnYN9kJmtNLO1Zra2sXHQE5GMuObthHD0TdTDaEREMr1YfKk/eLOZPQuUA78dYrVUt+smNzn9HnCDcy5mlv7uXufcXcBdANXV1UfdbLWjbjOlQGTaUCc1IiLj3xH3IOqc+9PQSwHeGcDshPFZDDyLqAbu95PAZOBiM4s65x470riOxMHaTZQC5bPU66iISJBdSb8ALDSzecAe4DLgo4kLOOcOPRjMzH4K/DroJAAQ2/catW4yc6ZVBr0pEZExL7AmM865KHAdXmugzcADzrmNZnaNmV0T1HYzkd+yndfdDGZPKs5mGCIiY0KgD5dxzq0GVidNS3lh2Dn3ySBjSdgQ5Z072Bt5F4WR8KhsUkRkLMu9RvRt9RTGu2gr1ePqRUQgFxNB02sAxCv1nGIREcjBRNBVvwWAguknZjkSEZGx4Zh/AP2R6tyzmagrYsr0OUMvLCKSA3LujCDe9Brb3QyOm1ya7VBERMaEnEsEha3bed3N5LhKNR0VEYFcSwS9HZT1NLA3MpuSgpyrFRMRSSm3EkFzDQCdE47PciAiImNHbiWCpm0AuEo9p1hEpF9O1Y/0NWwh5IySGbqHQESkX04lgq76Lex3U5k1ZWK2QxERGTNyrmpou5vJXLUYEhE5ZPyfEdy6EDq8p2BOAC4IAz86DkqmwqptWQ1NRGQsGP9nBB1pHoWcbrqISI4Z/4lAREQGpUQgIpLjlAhERHKcEoGISI4b94mgmYojmi4ikmvGffPR6u7bcSmmG/DGaAcjIjIGjfszgpkVRUc0XUQk14z7RLBq2SKKIuHDphVFwqxatihLEYmIjC3jvmpoxdIqAG59ait1LV3MrChi1bJFh6aLiOS6cZ8IwEsGKvhFRFIb91VDIiIyOCUCEZEcp0QgIpLjlAhERHKcEoGISI5TIhARyXFKBCIiOU6JQEQkxykRiIjkOCUCEZEcF2giMLOLzGyrmdWY2Y0p5l9hZq/4r+fM7K1BxiMiIgMFlgjMLAz8AFgOLAYuN7PFSYu9AfyNc+5U4JvAXUHFIyIiqQV5RnAmUOOce9051wvcD1ySuIBz7jnn3AF/9HlgVoDxiIhICkEmgipgd8J4rT8tnU8DT6aaYWYrzWytma1tbGwcwRBFRCTIRGAppqV6aiRmdj5eIrgh1Xzn3F3OuWrnXPWUKVNGMEQREQnyeQS1wOyE8VlAXfJCZnYq8CNguXOuOcB4REQkhSDPCF4AFprZPDPLBy4DnkhcwMzmAI8AVzrnXgswFhERSSOwMwLnXNTMrgOeAsLA3c65jWZ2jT//TuBrQCVwu5kBRJ1z1UHFJCIiA5lzKavtx6zq6mq3du3abIchInJMMbN16Q60c+KZxSIifX191NbW0t3dne1QAlVYWMisWbOIRCIZr6NEICI5oba2lrKyMubOnYtfFT3uOOdobm6mtraWefPmZbye+hoSkZzQ3d1NZWXluE0CAGZGZWXlEZ/1KBGISM4Yz0mg33C+oxKBiEiOUyIQEUnhsZf2cNYtf2Dejb/hrFv+wGMv7Tmqz2tpaeH2228/4vUuvvhiWlpajmrbQ1EiEBFJ8thLe7jpkQ3saenCAXtaurjpkQ1HlQzSJYJYLDboeqtXr6aiomLY282EWg2JSM75xn9uZFPdwbTzX9rVQm8sfti0rr4Yf//QK/xyza6U6yyeOYGvv//ktJ954403sn37dpYsWUIkEqG0tJQZM2awfv16Nm3axIoVK9i9ezfd3d186UtfYuXKlQDMnTuXtWvX0t7ezvLlyzn77LN57rnnqKqq4vHHH6eoqGgYe+BwOiMQEUmSnASGmp6JW265hfnz57N+/XpuvfVW1qxZw7e+9S02bdoEwN133826detYu3Ytt912G83NAwzmClgAAAqtSURBVLte27ZtG9deey0bN26koqKChx9+eNjxJNIZgYjknMGO3AHOuuUP7GnpGjC9qqKIX139jhGJ4cwzzzysrf9tt93Go48+CsDu3bvZtm0blZWVh60zb948lixZAsDpp5/Ojh07RiQWnRGIiCRZtWwRRZHwYdOKImFWLVs0YtsoKSk5NPzHP/6RZ555hr/85S+8/PLLLF26NOW9AAUFBYeGw+Ew0Wh0RGLRGYGISJIVS71naN361FbqWrqYWVHEqmWLDk0fjrKyMtra2lLOa21tZeLEiRQXF7Nlyxaef/75YW9nOJQIRERSWLG06qgK/mSVlZWcddZZnHLKKRQVFTFt2rRD8y666CLuvPNOTj31VBYtWsTb3/72EdtuJtT7qIjkhM2bN3PSSSdlO4xRkeq7Dtb7qK4RiIjkOCUCEZEcp0QgIpLjlAhERHKcEoGISI5TIhARyXG6j0BEJNmtC6Fj38DpJVNh1bZhfWRLSwu/+MUv+PznP3/E637ve99j5cqVFBcXD2vbQ9EZgYhIslRJYLDpGRju8wjASwSdnZ3D3vZQdEYgIrnnyRth74bhrfuT96aePv0tsPyWtKsldkN94YUXMnXqVB544AF6enq49NJL+cY3vkFHRwcf/vCHqa2tJRaL8dWvfpWGhgbq6uo4//zzmTx5Ms8+++zw4h6EEoGIyCi45ZZbePXVV1m/fj1PP/00Dz30EGvWrME5xwc+8AH+/Oc/09jYyMyZM/nNb34DeH0QlZeX893vfpdnn32WyZMnBxKbEoGI5J5BjtwBuLk8/byrfnPUm3/66ad5+umnWbp0KQDt7e1s27aNc845h+uvv54bbriB973vfZxzzjlHva1MKBGIiIwy5xw33XQTV1999YB569atY/Xq1dx000285z3v4Wtf+1rg8ehisYhIspKpRzY9A4ndUC9btoy7776b9vZ2APbs2cO+ffuoq6ujuLiYj33sY1x//fW8+OKLA9YNgs4IRESSDbOJ6GASu6Fevnw5H/3oR3nHO7ynnZWWlnLfffdRU1PDqlWrCIVCRCIR7rjjDgBWrlzJ8uXLmTFjRiAXi9UNtYjkBHVDrW6oRUQkDSUCEZEcp0QgIjnjWKsKH47hfEclAhHJCYWFhTQ3N4/rZOCco7m5mcLCwiNaT62GRCQnzJo1i9raWhobG7MdSqAKCwuZNWvWEa2jRCAiOSESiTBv3rxshzEmBVo1ZGYXmdlWM6sxsxtTzDczu82f/4qZnRZkPCIiMlBgicDMwsAPgOXAYuByM1uctNhyYKH/WgncEVQ8IiKSWpBnBGcCNc65151zvcD9wCVJy1wC3OM8zwMVZjYjwJhERCRJkNcIqoDdCeO1wNsyWKYKqE9cyMxW4p0xALSb2dZhxjQZaBrmuqNhrMcHYz9GxXd0FN/RGcvxHZduRpCJwFJMS263lckyOOfuAu466oDM1qa7xXosGOvxwdiPUfEdHcV3dMZ6fOkEWTVUC8xOGJ8F1A1jGRERCVCQieAFYKGZzTOzfOAy4ImkZZ4APu63Hno70Oqcq0/+IBERCU5gVUPOuaiZXQc8BYSBu51zG83sGn/+ncBq4GKgBugErgoqHt9RVy8FbKzHB2M/RsV3dBTf0Rnr8aV0zHVDLSIiI0t9DYmI5DglAhGRHDcuE8FY7trCzGab2bNmttnMNprZl1Isc56ZtZrZev8V/NOrD9/+DjPb4G97wOPgsrz/FiXsl/VmdtDMvpy0zKjvPzO728z2mdmrCdMmmdnvzGyb/z4xzbqD/l4DjO9WM9vi/w0fNbOKNOsO+nsIML6bzWxPwt/x4jTrZmv//Sohth1mtj7NuoHvv6PmnBtXL7wL09uB44F84GVgcdIyFwNP4t3H8Hbgr6MY3wzgNH+4DHgtRXznAb/O4j7cAUweZH7W9l+Kv/Ve4Lhs7z/gXOA04NWEaf8G3OgP3wh8O813GPT3GmB87wHy/OFvp4ovk99DgPHdDFyfwW8gK/svaf53gK9la/8d7Ws8nhGM6a4tnHP1zrkX/eE2YDPe3dTHkrHSNcgFwHbn3M4sbPswzrk/A/uTJl8C/Mwf/hmwIsWqmfxeA4nPOfe0cy7qjz6Pdx9PVqTZf5nI2v7rZ2YGfBj45Uhvd7SMx0SQrtuKI10mcGY2F1gK/DXF7HeY2ctm9qSZnTyqgXl3dz9tZuv87j2SjYn9h3dvSrp/vmzuv37TnH9fjP8+NcUyY2VffgrvLC+VoX4PQbrOr7q6O03V2ljYf+cADc65bWnmZ3P/ZWQ8JoIR69oiSGZWCjwMfNk5dzBp9ot41R1vBf4v8Nhoxgac5Zw7Da932GvN7Nyk+WNh/+UDHwAeTDE72/vvSIyFffkPQBT4eZpFhvo9BOUOYD6wBK//se+kWCbr+w+4nMHPBrK1/zI2HhPBmO/awswieEng5865R5LnO+cOOufa/eHVQMTMJo9WfM65Ov99H/Ao3ul3orHQNchy4EXnXEPyjGzvvwQN/VVm/vu+FMtk+7f4CeB9wBXOr9BOlsHvIRDOuQbnXMw5Fwd+mGa72d5/ecDfAr9Kt0y29t+RGI+JYEx3beHXJ/4Y2Oyc+26aZab7y2FmZ+L9nZpHKb4SMyvrH8a7oPhq0mJjoWuQtEdh2dx/SZ4APuEPfwJ4PMUymfxeA2FmFwE3AB9wznWmWSaT30NQ8SVed7o0zXaztv987wa2OOdqU83M5v47Itm+Wh3EC69Vy2t4rQn+wZ92DXCNP2x4D83ZDmwAqkcxtrPxTl1fAdb7r4uT4rsO2IjXAuJ54J2jGN/x/nZf9mMYU/vP334xXsFenjAtq/sPLynVA314R6mfBiqB3wPb/PdJ/rIzgdWD/V5HKb4avPr1/t/hncnxpfs9jFJ89/q/r1fwCvcZY2n/+dN/2v+7S1h21Pff0b7UxYSISI4bj1VDIiJyBJQIRERynBKBiEiOUyIQEclxSgQiIjlOiUAkYH5vqL/Odhwi6SgRiIjkOCUCEZ+ZfczM1vj9xv+HmYXNrN3MvmNmL5rZ781sir/sEjN7PqEv/4n+9AVm9ozf4d2LZjbf//hSM3vI7///5wl3Pt9iZpv8z/nfWfrqkuOUCEQAMzsJ+AheB2FLgBhwBVCC16fRacCfgK/7q9wD3OCcOxXv7tf+6T8HfuC8Du/eiXc3Kni9zH4ZWIx3t+lZZjYJr+uEk/3P+edgv6VIakoEIp4LgNOBF/wnTV2AV2DHebNDsfuAs82sHKhwzv3Jn/4z4Fy/T5kq59yjAM65bvdmHz5rnHO1zutAbT0wFzgIdAM/MrO/BVL29yMSNCUCEY8BP3POLfFfi5xzN6dYbrA+WVJ1idyvJ2E4hvdksCheT5QP4z205rdHGLPIiFAiEPH8HvigmU2FQ88bPg7vf+SD/jIfBf7bOdcKHDCzc/zpVwJ/ct5zJWrNbIX/GQVmVpxug/4zKcqd11X2l/H63RcZdXnZDkBkLHDObTKzf8R7klQIr5fJa4EO4GQzWwe04l1HAK9b6Tv9gv514Cp/+pXAf5jZP/mf8aFBNlsGPG5mhXhnE18Z4a8lkhH1PioyCDNrd86VZjsOkSCpakhEJMfpjEBEJMfpjEBEJMcpEYiI5DglAhGRHKdEICKS45QIRERy3P8HEajPolyqLYIAAAAASUVORK5CYII="/>

***

## 7.6 CNN 시각화하기

CNN을 구성하는 Conv layer가 입력으로 받은 이미지 데이터에서 보고있는 것을 visualize를 통해 확인해본다.



- 1번째 층의 가중치 시각화하기  

위의 예시에서 1번째 층의 weight는 (30, 1, 5, 5)의 형상이다. 채널 1, 5 \* 5 크기의 필터는 회색조 이미지로 시각화할 수 있다. 학습 전과 후의 weight를 비교하여 본다.



```python
import numpy as np
import matplotlib.pyplot as plt

def filter_show(filters, nx=8, margin=3, scale=10):
    """
    c.f. https://gist.github.com/aidiary/07d530d5e08011832b12#file-draw_weight-py
    """
    FN, C, FH, FW = filters.shape
    ny = int(np.ceil(FN / nx))

    fig = plt.figure()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(FN):
        ax = fig.add_subplot(ny, nx, i+1, xticks=[], yticks=[])
        ax.imshow(filters[i, 0], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()


network = SimpleConvNet()
# 무작위(랜덤) 초기화 후의 가중치
filter_show(network.params['W1'])

# 학습된 가중치
network.load_params("params.pkl")
filter_show(network.params['W1'])
```

<img src="data:image/png;base64,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"/>

<img src="data:image/png;base64,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"/>

***

학습 전 필터는 무작위로 초기화되어 흑백의 규칙성이 없으나, 학습을 마친 필터는 규칙성이 생긴다. (덩어리 지거나, 흰색에서 검은색으로 점차 변화한다.)



이런 필터의 규칙성은 이미지에서 edge(색상이 바뀌는 경계선), blob(국소적으로 덩어리진 영역) 등을 보고 있다.



![7-20](/assets/images/7-20.png)



위처럼 필터는 edge나 blob 등의 원시적인 정보를 추출하여 다음 layer에게 전달한다.


- 층 깊이에 따른 추출 정보 변화  

1층 layer에서 edge, blob 같은 저수준 정보가 추출되고, 층이 깊어질수록 추출되는 정보는 더 추상화된다. 다음은 CNN의 한 종류인 AlexNet의 예시다.



![7-21](/assets/images/7-21.png)



edge + blog, texture, object parts, object classes 순으로 층이 깊어질수록 고급 정보를 인식함을 알 수 있다.


## 7.7 대표적인 CNN

여러 CNN 모델 중 CNN의 원조 격인 LeNet과 딥러닝을 주목받도록 이끈 AlexNet을 살펴본다.

- LeNet  
  <br>
  LeNet은 CNN이라는 개념을 최초로 개발한 Yann LeCun이 1998년에 개발한 구조이다.  
  ![7-22](/assets/images/7-22.png)  
  Conv layer와 Pooling layer (정확히는 단순이 원소를 줄이는 subsampling layer)를 반복하고, 마지막에 fully-connected layer를 거치며 결과를 출력한다.  
    - activation function으로 sigmoid function을 사용한다. 현재는 주로 ReLU를 사용한다.  
    - subsampling으로 크기를 줄이지만, 현재는 max pooling이 주류이다.

<br>

- AlexNet  
  <br>
  AlexNet은 2012년 ImageNet 데이터 기반의 ILSVRC (ImageNet Large Scale Visual Recognition Challenge) 대회에서 우승한 구조이다. original 논문명이 'ImageNet Classification with Deep Convolutional Neural Networks'이고 논문 첫 저자가 Alex Khrizevsky라서 AlexNet으로 불린다.  
![7-23](/assets/images/7-23.png)
AlexNet은 LeNet에서 큰 구조는 바뀌지 않았지만 다음의 특징이 있다.  
    - activation function으로 ReLU 사용  
    - LRN (Local Response Normalization)이라는 국소적 정규화를 실시하는 계층을 이용한다.  
    - Drop-out을 사용한다.

<br>

> 네트워크 구성에는 LeNet과 AlexNet에 큰 차이가 없지만, 병렬 계산에 특화된 GPU가 보급되고, 빅데이터 접근성이 좋아지면서 딥러닝 성능이 크게 발전하게 되었다.  


# 7.8 정리

- CNN은 완전연결 네트워크에 Conv layer, Pooling layer를 새로 추가한다.  

- Conv layer와 Pooling layer는 im2col 함수를 이용해 효율적으로 구현한다.  

- CNN은 층이 깊어질수록 고급 정보를 추출한다.  

- 대표적인 CNN으로 LeNet과 AlexNet이 있다.  

- GPU와 빅 데이터가 딥러닝 발전을 불러왔다.

