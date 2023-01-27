---
title: "Baekjoon problem step2 conditional (조건문)"
excerpt: "Step2 조건문 풀이집입니다."
date: 2022-10-10
categories:
    - Algorithm
tags:
    - Baekjoon
last_modified_at: 2022-10-10
---
- Step2 조건문 풀이집입니다.

# Conditional, 조건문
if 등의 조건문 사용해보기 - [step 2 문제 리스트](https://www.acmicpc.net/step/4)  
문제 상황의 조건에 맞게 어떻게 동작할 지 배우는 단계입니다. 기초적인 logic을 통해 case를 나누고 코드를 구성합니다.  

## 2-1. 1330번, 두 수 비교하기  
[prob 1330](https://www.acmicpc.net/problem/1330) : 두 수가 주어졌을 때 > < == 로 크기를 비교한다.  
```python
a, b = map(int, input().split())
if a < b:
    print("<")
elif a == b:
    print("==")
else:
    print(">")
```
풀이 : 처음 if이후 추가 조건은 else if의 줄임말인 elif로 조건을 세분화한다.

## 2-2. 9498번, 시험 성적  
[prob 9498](https://www.acmicpc.net/problem/9498) : 시험 점수를 입력받아 10점 단위로 A, B, C, D, F 등급을 매긴다.  
```python
a = int(input())
if a >= 90:
    print("A")
elif a >= 80:
    print("B")
elif a >= 70:
    print("C")
elif a >= 60:
    print("D")
else:
    print("F")
```
풀이 : if와 elif 외에 나머지 경우는 else로 처리 가능하다.

## 2-3. 2753번, 윤년
[prob 2753](https://www.acmicpc.net/problem/2753) : 주어진 연도가 윤년인지 확인한다. 윤년은 연도가 4의 배수이면서, 100의 배수가 아닐 때 또는 400의 배수일 때이다.  
```python
a = int(input())
if (a % 4 == 0 and a % 100 != 0) or a % 400 == 0:
    print(1)
else:
    print(0)
```
풀이 : "a면서 b이다"는 and로, "a 또는 b이다"는 or로 조건을 구성해준다.

## 2-4. 14681번, 사분면 고르기
[prob 14681](https://www.acmicpc.net/problem/14681) : (x, y) 좌표가 주어질 때 어느 사분면에 속하는지 확인한다. x, y좌표는 0이 아니라고 가정한다.
```python
a = int(input())
b = int(input())
if a > 0 and b > 0:
    print(1)
elif a < 0 and b > 0:
    print(2)
elif a < 0 and b < 0:
    print(3)
else:
    print(4)
```
풀이 : 사분면 조건에 맞게 x 조건, y 조건을 and로 구성해준다.

## 2-5. 2884번, 알람 시계
[prob 2884](https://www.acmicpc.net/problem/2884) : 원래 알람 시간보다 45분 일찍 울릴 수 있도록 알람 시계를 설정한다.
```python
a, b = map(int, input().split())
if b < 45:
    b = 60 - (45 - b)
    if a == 0:
        a = 23
    else:
        a -= 1
else:
    b -= 45
print(a, b)
```
풀이 : 시가 바뀌는 경우와 바뀌지 않는 경우를 구분하면 쉽다.

## 2-6. 2525번, 오븐 시계
[prob 2525](https://www.acmicpc.net/problem/2525) : 훈제 요리 시작 시간과 필요 시간이 주어졌을 때, 요리 완성 시간을 구한다.
```python
a, b = map(int, input().split())
c = int(input())
h = int(c / 60)
a = (a + h) % 24
m = c % 60
if b + m > 59:
    b = b + m - 60
    if a == 23:
        a = 0
    else:
        a += 1
else:
    b += m
print(a, b)
```
풀이 : 추가되는 c분을 h시간 m분으로 바꾸고 더했을 때 시간, 분이 각각 24시, 60분을 넘어가는 케이스를 고려해주면 된다.

## 2-7. 2480번, 주사위 세개
[prob 2480](https://www.acmicpc.net/problem/2480) : 1에서부터 6까지의 눈을 가진 3개의 주사위를 던져서 다음과 같은 규칙에 따라 상금을 받는 게임이 있다.  
1. 같은 눈이 3개가 나오면 10,000원+(같은 눈)×1,000원의 상금을 받게 된다. 
2. 같은 눈이 2개만 나오는 경우에는 1,000원+(같은 눈)×100원의 상금을 받게 된다. 
3. 모두 다른 눈이 나오는 경우에는 (그 중 가장 큰 눈)×100원의 상금을 받게 된다. 
   
3개 주사위의 나온 눈이 주어질 때 상금을 계산한다.  
```python
a, b, c = map(int, input().split())
if a == b and b == c:
    print(10000 + a * 1000)
elif a == b or b == c or a == c:
    if a == b or a == c:
        print(1000 + a * 100)
    else:
        print(1000 + b * 100)
else:
    print(max(a, b, c) * 100)
```
풀이 : 같은 눈이 2개일 경우 같은 눈이 무엇인지 잘 구별하도록 하는 것이 중요하다.

# step2 : 조건문 요약
if, elif, else로 case만 잘 나누면 크게 어렵지 않은 것 같다. 각 조건을 어떤 순서로 배치해야 더 효율적일지도 고려해보면 좋을 것 같다.