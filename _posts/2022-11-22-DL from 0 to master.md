---
title: "[책] 밑바닥부터 시작하는 딥러닝"
excerpt: "밑바닥부터 시작하는 딥러닝 정리"
date: 2022-11-22
categories:
    - AI
    - Books
tags:
    - DL
last_modified_at: 2022-11-23
---
- 딥러닝 입문서 '밑바닥부터 시작하는 딥러닝'을 정리한 내용입니다.


# 책 배경
파이썬과 numpy, matplotlib를 통해 딥러닝을 구현하는 과정을 차근차근 다룬 책입니다. 2편은 자연어 처리, 3편은 딥러닝 프레임워크를 다룹니다.

책에서 다루는 코드들은 <https://github.com/WegraLee/deep-learning-from-scratch>에서 다운받아서 활용할 수 있습니다.

***

# 책 목차
목차별로 나눠서 jupyter notebook에서 예제 python 파일을 실행한 코드와 결과값, 설명을 정리해두었습니다.

## 1. 헬로 파이썬
파이썬에 대한 기초적인 설명, numpy, matplotlib에 대해 배웁니다.

[ch.1 헬로 파이썬](../../DL-from-0-to-master-ch.1)


## 2. 퍼셉트론
논리 회로, 퍼셉트론의 구조, 단점, 다층 퍼셉트론에 대해 배웁니다.

[ch.2 퍼셉트론](../../DL-from-0-to-master-ch.2)


## 3. 신경망
활성화 함수, 다차원 배열 계산, 다층 신경망 설계, MNIST 데이터 학습 신경망 구현을 배웁니다.

[ch.3 신경망](../../DL-from-0-to-master-ch.3)


## 4. 신경망 학습
loss function, 경사 하강법, 학습 알고리즘 구현을 배웁니다.

[ch.4 신경망 학습](../../DL-from-0-to-master-ch.4)


## 5. 오차역전파법
back-propagation, 각 layer를 class로 모듈화해서 구현하는 방법을 배웁니다.

[ch.5 오차역전파법](../../DL-from-0-to-master-ch.5)


## 6. 학습 관련 기술들
parameter 갱신하는 optimizer, 가중치 초깃값, batch normalization, overfitting, weight decay, droupout, hyper-parameter 값 찾는 법을 배웁니다. 

[ch.6 학습 관련 기술들](../../DL-from-0-to-master-ch.6)


## 7. 합성곱 신경망(CNN)
합성곱 계층, 풀링 계층의 구조 및 구현, CNN 구조 및 구현, LeNet, AlexNet에 대해 배웁니다. 

[ch.7 합성곱 신경망(CNN)](../../DL-from-0-to-master-ch.7)


## 8. 딥러닝
층을 깊게 하는 이유, 대표적인 딥러닝 모델들, 딥러닝 고속화, 딥러닝 활용 분야들을 배웁니다.

[ch.8 딥러닝](../../DL-from-0-to-master-ch.8)

***

# 책 소감  
딥러닝을 이해하는데 필요한 기초적인 내용들을 그림, 코드로 자세하게 서술해두어 이해하기 편하다. 하지만 책에 실린 기계학습 기법들, 딥러닝 모델들의 구조, 작동방식 등은 자세한 서술은 되어 있지 않아 추가로 공부가 필요해 보인다. 입문서로서 좋은 책인 것 같다.