---
title: "밑바닥부터 시작하는 딥러닝 ch.8"
excerpt: "ch.8 딥러닝"
date: 2022-12-14
categories:
    - AI
tags:
    - DL
use_math: true
last_modified_at: 2022-12-14
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


# 8. 딥러닝  

- 딥러닝은 층을 깊게한 심층 신경망으로, 이번 장에서 딥러닝의 특징, 과제, 가능성을 살펴본다.  

- 참고 : <https://data-scientist-brian-kim.tistory.com/87>, <https://velog.io/@dscwinterstudy/2020-01-28-1401-%EC%9E%91%EC%84%B1%EB%90%A8-tfk5xgv65x>



## 8.1 더 깊게

그동안 배웠던 layer, 학습 기술들을 집약해 심층 신경망을 만들어 MNIST 데이터를 학습시켜 본다.



아래의 그림과 같이 구성된 CNN을 만들고자 한다.  



![8-1](/assets/images/8-1.png)

(위의 구조는 다음 절에서 설명할 VGG 신경망을 참고했다.)  

위의 신경망은 다음의 특징들을 가진다.  

- Conv layer에서 3 \* 3 크기의 작은 필터를 사용한다.  
- activation function으로 ReLU를 사용한다.  
- 완전연결 계층 (Affine) 뒤에 Drop-out layer 사용했다.  
- Optimizer로 Adam을 사용한다.  
- weight 초기값으로 'He 초깃값' 사용한다. (ReLU 사용시 He 초깃값이 효율적이다.)



위의 신경망은 99.38%의 정확도를 가진다. 잘못 인식한 이미지들은 사람도 인식 오류를 저지르는 애매한 숫자 데이터들이다. (0처럼 보이는 6, 3처럼 보이는 1 등) 

***

- 정확도를 더 높이려면  
웹사이트 [What is the class of this image?](https://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html)에서는 다양한 데이터셋에 대해 논문 등에서 발표한 기법들의 정확도 순위를 정리해두었다. 



순위 상위권은 대부분 CNN을 기반으로 한 기법들이다.  다만 MNIST 데이터셋은 손글씨 숫자로 비교적 단순하기 때문에 층이 깊지 않은 CNN들이다. (추후 소개될 대규모 일반 사물 인식은 복잡하기 때문에 층을 깊게 해야 정확도를 끌어올릴 수 있다.)



사이트에 나온 상위 기법들은 정확도를 높일 'ensemble learning', 'learning rate decay', data augmentation' 등의 기법들을 사용한다. 이중에서 앞서 살펴보지 못한 data augmentation을 살펴본다.

***

- 데이터 확장 data augmentation  
**데이터 확장 data augmentation**은 training image data를 알고리즘으로 변형을 주어 이미지 개수를 늘리는 방식이다. 이는 데이터 개수가 적을 때 특히 효과적이다. 이미지 변형은 다음의 방식들이 존재한다.  
    - 이미지 회전 (rotate)  
    - 이미지 이동  
    - 이미지 일부 잘라내기 (crop)  
    - 이미지 좌우/상하 반전 (flip) : 이미지 대칭성을 고려하지 않는 경우에 사용  
    - 이미지 확대/축소 (scaling)  
    - 이미지 밝기 변화 (lighting condition)  

    
***

- 층을 깊게 하는 이유  
층을 깊게하는 것이 왜 중요한가에 대한 이론적인 근거는 부족한 것이 사실이다. 하지만 다음의 연구, 실험 결과로 직관적인 설명이 존재한다.



ILSVRC (ImageNet Large Scale Visual Recognition Challenge) 대규모 이미지 인식 대회에서 최근 상위권을 차지한 기법 대부분이 딥러닝 기반으로, 층을 깊게하는 방향으로 가고 있다. 층을 깊게 할수록 정확도가 좋아짐을 확인할 수 있다.



**층을 깊게 구성할 때는 신경망의 parameter 수가 줄어드는 이점이 있다.** 층을 깊게 할 때는 깊지 않은 경우보다 적은 parameter로 같거나 그 이상의 표현력을 달성할 수 있다. 다음의 예시를 살펴본다.  



![8-2](/assets/images/8-2.png)


5 \* 5 Conv 연산 1회와 3 \* 3 Conv 연산 2회는 둘다 출력 데이터 한 곳이 입력 데이터 5 \* 5 범위에서 구해짐을 알 수 있다. 전자는 parameter 수가 25개인 반면, 후자는 18개이다. 이처럼 같은 범위를 커버하더라도 층을 늘릴수록 parameter 수가 적어진다.  



> 작은 필터를 겹쳐 layer를 깊게하면 parameter 수를 줄이고 넓은 **수용 영역 receptive field**를 소화할 수 있다. (수용 영역은 뉴런에 변화를 일으키는 국소적 공간 영역을 말한다.) 또한, 층을 거듭하며 activation function을 Conv layer 사이에 끼우면서 신경망의 표현력이 개선된다. (activation function이 신경망에 '비선형' 힘을 가하고, 비선형 함수가 겹쳐지면서 더 복잡한 것을 표현할 수 있게 되기 때문이다.)

***

**또한 층을 깊게 구성하면 학습 데이터 양을 줄여 학습을 고속으로 수행할 수 있다.** (신경망을 깊게 하면 문제를 계층적으로 분해해서 효율적으로 학습한다.) 앞에서 봤던 것처럼 층이 깊어질수록 고도화된 정보를 얻기 때문에 각 층이 학습하는 문제를 단순한 문제로 분해하게 된다. 이는 층이 얕은 신경망보다 학습 데이터를 적게 사용하고 더 좋은 성과를 낼 수 있다.

***

정리하자면, 층을 깊게 하는 것의 이점은 아래의 것들이 있다.  

1. 수용 영역을 잘 커버해서 parameter 수를 줄인다.  
2. 정보를 단계적으로 얻어 고도화된 패턴을 학습하는데 효율적이라 학습데이터를 줄일 수 있다.  



> 층을 깊게하는 경향은 빅 데이터, 컴퓨터 연산 능력 등 새로운 기술, 환경이 뒷받침되어 나타날 수 있었음을 이해해야 한다.


## 8.2 딥러닝의 초기 역사  

딥러닝이 주목받게 된 계기는 2012년 ILSVRC (ImageNet Large Scale Visual Recognition Challenge) 대회에서 AlexNet이 압도적으로 우승하면서이다. 그러므로 ILSVRC 대회를 축으로 최근 딥러닝 트렌드를 살펴본다.

***

- 이미지넷 ImageNet  
**이미지넷 ImageNet**은 앞서 배운 딥러닝 모델들 (AlexNet, LeNet) 처럼 Net이 붙었지만 딥러닝 모델이 아니라 이미지 데이터셋이다. 다양한 클래스의 레이블이 달린 100만 장이 넘는 이미지 데이터셋이다.  



ILSVRC가 이 거대 데이터셋을 사용해 이미지 인식 기술을 경쟁하는 대회이다. (ILSVRC 시험 항목 중 하나인 **분류 Classification**은 1000개의 클래스를 잘 분류하는지를 겨룬다.) ILSVRC에서 좋은 성과를 거둔 유명한 모델들 (VGG, GoogLeNet, ResNet을 살펴본다.

***

- VGG  
**VGG Visual Geometry Group**는 Conv layer와 Pooling layer로 구성된 기본적인 CNN이다.  

![8-3](/assets/images/8-3.png)

비중 있는 층 (Conv layer, 완전연결 layer)을 모두 16층 (혹은 19층)으로 심화한 것이 특징이다. (층 깊이에 따라 'VGG16'과 'VGG19'로 구분하기도 한다.)


![8-4](/assets/images/8-4.png)


VGG는 3 \* 3 크기의 작은 필터를 사용한 Conv layer를 연속으로 거치고 2~4회 연속으로 Pooling layer를 두어 크기를 절반으로 줄이는 처리를 반복한다. 그리고 마지막에 Fully-connected layer를 통과시켜 결과를 출력한다.  

> VGG는 2014년 대회에서 GoogLeNet에 밀려 2위를 차지했다. VGG는 구성이 간단해 응용하기 좋다.

***

- GoogLeNet  

GoogLeNet의 구조는 다음과 같다.  

![8-5](/assets/images/8-5.png)

GoogLeNet은 세로 방향 깊이뿐만 아니라, 가로 방향도 깊다는 점이 특징이다.


GoogLeNet에는 가로 방향에 '폭'이 있다. 이를 인셉션 구조라고 한다.  

![8-6](/assets/images/8-6.png)

인셉션 구조는 크기가 다른 필터와 pooling 여러 개를 적용한 결과를 결합한다. 그리고 이 인셉션 구조를 하나의 블록 (구성 요소)로 사용하는 것이 GoogLeNet의 특징이다. 



GoogLeNet에선 1 \* 1 크기의 필터를 사용한 Conv layer를 많은 곳에 사용한다. (1 \* 1 Conv layer는 채널 쪽으로 크기를 줄이는 것으로, parameter 제거와 고속 처리에 기여한다.)

***

- ResNet, Residual Network  

**ResNet Residual Network**는 microsoft에서 개발한 네트워크다.  

딥러닝 학습에서 층을 지나치게 깊게하면 학습이 잘 되지 않고, 성능이 떨어진다. ResNet은 이를 막기 위해 **스킵 연결 skip connection**을 사용한다. (Skip connection을 통해 층의 깊이에 비례해 성능을 향상시킬 수 있게 한다.)  

![8-7](/assets/images/8-7.png)  

Skip connection은 입력 데이터가 Conv layer를 건너뛰어 출력에 바로 더하는 구조를 말한다. 이를 통해 역전파 때 신호 감쇠를 막아준다.  

> skip connection은 입력 데이터를 그대로 흘리는 것으로, 역전파 때도 상류의 기울기를 그대로 하류로 보낸다. 층을 깊게 할수록 기울기가 작아지는 gradient vanishing 문제를 줄여준다.


ResNet은 VGG 신경망을 기반으로 skip connection을 도입해 층을 깊게 했다.  

![8-8](/assets/images/8-8.png)

위와 같이 Conv layer를 2개 층마다 건너뛰면서 층을 깊게 한다.  



> **전이 학습 transfer learning**을 통해 ImageNet이 제공하는 거대한 데이터셋으로 학습한 가중치 값들을 활용할 수 있다. transfer learning은 학습된 가중치(혹은 일부)를 다른 신경망에 복사한 후, 새로운 데이터셋을 대상으로 **재학습 fine tuning**을 수행하는 것을 말한다. transfer learning은 보유한 데이터셋이 적을 때 특히 유용하다.



## 8.3 더 빠르게 (딥러닝 고속화)  

딥러닝은 빅 데이터와 네트워크의 발전으로 대량의 연산을 수행해야 한다. 과거에는 주로 CPU가 연산을 담당했지만, 최근 딥러닝 프레임워크 대부분이 **GPU Graphics Processing Unit**을 활용해 대량 연산을 고속으로 처리한다. (최근 프레임워크는 복수의 GPU와 여러 기기로 분산 수행을 한다.)



딥러닝에서 어떤 처리가 얼만큼 시간이 소요되는지를 AlexNet의 forward 처리를 통해 알아본다. CPU에선 전체의 89%, GPU에선 전체의 95%의 시간이 Conv layer에서 소요된다. (학습에서도 Conv layer에서 많은 시간이 소요된다. Conv layer는 '단일 곱셈 누산 FMA' 연산이 처리된다.) 즉, 딥러닝 고속화는 FMA를 어떻게 고속으로 효율적으로 처리하는지의 문제이다.

***

- GPU를 활용한 고속화  

GPU는 본래 그래픽 전용 보드에 이용해왔으나, 최근에 그래픽 처리말고도 범용 수치 연산에 이용한다. **GPU computing**이 GPU의 병렬 수치 연산 능력을 활용하는 방식이다. (CPU는 연속적인 복잡한 계산을 잘 처리하고, GPU는 대량 병렬 연산을 잘 처리한다.)



GPU와 딥러닝 최적화 라이브러리 (cuDNN 등)을 사용하면 CPU만 사용할 때보다 훨씬 학습 시간을 단축할 수 있다. GPU는 NVIDIA, AMD 두 회사에서 제공하지만, NVIDIA가 딥러닝에 적합하다. 대부분 딥러닝 프레임워크가 **CUDA Compute Unified Device Architecture**라는 NVIDIA의 GPU 컴퓨팅용 통합 개발 환경을 사용하기 때문이다. (cuDNN도 CUDA 위에서 작동하는 딥러닝 최적화 라이브러리다.)  

> 이전에 배웠던 im2col 연산으로 텐서를 행렬 내적으로 변환해 큰 덩어리를 한번에 계산하는 방식이 GPU를 잘 활용하는 방식이다.

***

- 분산 학습  

GPU가 딥러닝 연산을 가속시키지만, 신경망을 만들려면 여러 시행착오를 거치기에 최대한 학습 시간을 줄여야 한다. 이를 위해 딥러닝 학습을 **수평 확장 scale out**하게 된다.  

딥러닝 계산 고속화를 위해 다수의 GPU와 기기로 계산을 분산한다. (구글의 tensor-flow와 마이크로소프트의 CNTK Computational Network Toolkit이 분산학습에 중점을 두고 개발하고 있다.) Tensor-flow를 기준으로 GPU 100개를 사용하면 GPU 1개일 때 비해 56배 학습 속도가 빨라졌다. 



분산 학습을 하려면 계산을 어떻게 분산시키는지, 컴퓨터 사이 통신과 데이터 동기화 같은 문제들을 해결해야 한다. 이런 어려운 문제들은 직접 해결하기보다 그냥 뛰어난 프레임워크에 맡기는 것이 좋다.

***

- 연산 정밀도와 비트 줄이기  
  <br>
계산 능력 외에 '메모리 용량'과 '버스 대역폭' 등이 딥러닝 고속화를 위해 해결해야 할 문제이다.  
    - 메모리 용량 : 대량의 parameter, 중간 데이터를 메모리에 저장해야 한다.  
    - 버스 대역폭 : GPU, CPU의 버스를 흐르는 데이터가 많아져 한계를 넘으면 병목된다.

    

위의 문제를 위해 주고받는 데이터의 비트 수를 최소로 만들어야 한다. 컴퓨터는 주로 64bit나 32bit 부동소수점 수로 실수를 표현하지만, 신경망은 노이즈에 어느정도 robust하기 때문에 bit수를 줄여도 크게 문제가 없다. (16 비트 반정밀도만 사용해도 학습이 문제가 없다고 알려져 있다.) 또한 딥러닝 비트 수를 줄이는 연구가 계속 진행 중에 있다.


## 8.4 딥러닝의 활용  

딥러닝은 이미지, 음성, 자연어 처리 등 수많은 분야에서 뛰어난 성능을 보인다. 이중에서 컴퓨터 비전 분야를 중심으로 몇 가지를 알아본다.  

***

- 사물 검출 Object detection  

**사물 검출 object detection**은 이미지 속에 담긴 사물의 위치와 클래스를 알아내는 기술이다. 따라서 object detection은 object recognition보다 어려운 문제다. 



CNN을 사용해 object detection을 수행하는 기술 중 **R-CNN Regions with Convolutional Neural Network**를 소개한다.  

![8-9](/assets/images/8-9.png)

R-CNN은 크게 후보 영역 추출과 CNN 특징 계산 두 가지 처리로 이뤄진다. 후보 영역 추출은 논문에선 Selective Search 기법을 사용했고, 최근엔 후보 영역 추출까지 CNN으로 처리하는 **Faster R-CNN**기법도 등장했다. (Faster R-CNN은 모든 일을 하나의 CNN에서 처리하기 때문에 빠르다.)

***

- 분할 segmentation  

**분할 segmentation**은 이미지를 픽셀 수준에서 분류하는 문제이다. 픽셀 단위로 객체마다 채색된 supervised data를 사용해 학습한다.  

![8-10](/assets/images/8-10.png)

(왼쪽이 입력 이미지, 오른쪽이 지도용 이미지)


신경망으로 segmentation을 풀기 위해 **FCN Fully Convolutional Network**가 고안되었다. 단 한번의 forward 처리로 모든 픽셀의 클래스를 분류하는 기법이다.  



![8-11](/assets/images/8-11.png)


FCN은 'Conv layer만으로 구성된 네트워크'이다. 일반적 CNN이 Fully-connected layer를 사용하지만, FCN은 완전연결 계층을 같은 기능을 하는 Conv layer로 바꾼다. 이를 통해 완전연결 계층이 중간 데이터 공간 볼륨을 1차원으로 변환해 한 줄로 늘어선 노드들이 처리한 것과 달리 공간 볼륨을 유지한 채 마지막 출력까지 처리 가능하다.

> 32 \* 10 \* 10 데이터에 대해 32 \* 10 \* 10인 Conv layer 100개로 출력 노드가 100개인 완전연결 계층을 대체할 수 있다. 



FCN은 마지막에 공간 크기를 확대하는 처리를 수행한다. 이는 **이중 선형 보간 bilinear interpolation**에 의산 선형 확대로, **역합성곱 deconvolution**연산으로 구현한다.

***

- 사진 캡션 생성  

컴퓨터 비전과 자연어를 융합한 연구도 있다. 사진이 주어질 때, 사진을 설명하는 글 (사진 캡션)을 자동으로 생성하는 연구다. 사진 캡션 생성 모델로는 **NIC Neural Image Caption** 모델이 대표적이다. NIC는 심층 CNN과 자연어를 다루는 **순환 신경망 Recurrent Neural Network, RNN**으로 구성된다. (CNN으로 사진에서 특징을 추출해 특징을 RNN에 넘겨 RNN에서 텍스트를 생성한다.)

> RNN은 순환적 관계를 갖는 신경망으로 자연어나 시계열 데이터 등의 연속된 데이터를 다룰 때 많이 활용한다.  
>
> 여러 종류의 정보를 조합하고 처리하는 것을 **멀티모달 처리 multimodal processing**이라고 한다.


## 8.5 딥러닝의 미래  

딥러닝의 가능성과 미래를 느낄만한 연구를 소개한다.  

***

- 이미지 스타일(화풍) 변환  

두 이미지 (콘텐츠 이미지, 스타일 이미지)를 조합해 새로운 그림을 그려주는 연구다. 아래는 A Neural Algorithm of Artistic Style 논문을 구현해 적용한 예이다.  

![8-12](/assets/images/8-12.png)

이 기술은 네트워크의 중간 데이터가 콘텐츠 이미지의 중간 데이터와 비슷해지도록 학습한다. 또한, 스타일 이미지의 화풍을 흡수하기 위해 '스타일 행렬'이라는 개념을 도입하고 이 스타일 행렬의 오차를 줄이도록 학습한다. 

***

- 이미지 생성  

이미지를 주고 스타일을 바꾸는 것뿐만 아니라, 이미지를 생성하는 것도 가능하다. **DCGAN Deep Convolutional Generative Adversarial Network**기법은 이미지를 생성하는 과정을 모델화한다. **생성자 Generator**와 **식별자 Discriminator**로 불리는 2개의 신경망을 구성하고 서로 경쟁시켜가며 학습한다. 이를 통해 학습이 끝난 후엔 진짜와 착각할 정도의 이미지를 그려낸다. 

***

- 자율 주행  

SegNet은 CNN 기반 신경망으로 입력 이미지를 segmentation 한여 아래와 같이 주위 환경을 인식한다.  

![8-13](/assets/images/8-13.png)

***

- Deep Q-Network (DQN, 강화학습)  

에이전트가 환경에 맞게 행동을 선택하고, 환경에서 보상을 받고 보상을 통해 행동을 더 나은 보상을 받도록 바로 잡는 것이 **강화학습 reinforcement learning**이라고 한다.  

![8-14](/assets/images/8-14.png)


딥러닝을 사용한 강화학습 중 **Deep Q-Network DQN**이 존재한다. 이는 Q learning이라는 강화학습 알고리즘을 기초로 한다. Q learning은 최적 행동 가치 함수로 최적인 활동을 정한다. 이 함수를 CNN으로 비슷하게 흉내내 사용하는 것이 DQN이다.



DQN은 이전의 비디오 게임 학습에서 상태를 미리 추출해야 학습이 가능한 것과 달리 게임 영상만 주어지면 학습이 가능하다는 장점이 있다. 


## 8.6 정리

- 다양한 문제에서 신경망을 더 깊게 하여 성능을 개선할 수 있다.  

- 이미지 인식 기술 대회 ILSVRC에선 딥러닝 기반 기술이 상위권을 독점하고, 깊이가 더 깊어지는 추세다.  

- ILSVRC에서 유명한 신경망으로 VGG, GoogLeNet, ResNet이 있다.  

- GPU와 분산 학습, 비트 정밀도 감소 등으로 딥러닝을 고속화할 수 있다.  

- 딥러닝은 image recognition뿐 아니라 image detection, segmentation에 이용할 수 있다.  

- 딥러닝 응용분야로 사진 캡션 생성, 이미지 생성, 강화학습 등이 있다.

