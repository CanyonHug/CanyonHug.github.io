---
title: "Baekjoon problem step4 1-D Array (1차원 배열)"
excerpt: "Step4 1차원 배열 풀이집입니다."
date: 2022-10-11
categories:
    - Algorithm
tags:
    - Baekjoon
last_modified_at: 2022-10-11
---
- Step4 1차원 배열 풀이집입니다.

# 1-D Array, 1차원 배열
배열을 사용해보기 - [step 4 문제 리스트](https://www.acmicpc.net/step/6)  
데이터를 배열에 저장하고 꺼내쓰는 방법을 배웁니다.  

## 4-1. 10818번, 최소, 최대
[prob 10818](https://www.acmicpc.net/problem/10818) : N개의 정수가 주어질 때, 최솟값과 최댓값을 구한다. 
```python
n = int(input())
a = list(map(int, input().split()))
print(min(a), max(a))
```
풀이 : list()로 리스트를 만든 후, min()과 max()함수를 써서 쉽게 해결가능하다. min max의 매개변수로는 iterable 자료형만 들어올 수 있다.

## 4-2. 2562번, 최댓값
[prob 2562](https://www.acmicpc.net/problem/2562) : 9개의 서로 다른 자연수가 주어질 때, 이들 중 최댓값을 찾고 그 최댓값이 몇 번째 수인지를 구한다.
```python
max_n = 0
index = 0
for i in range(9):
    a = int(input())
    if a > max_n:
        max_n = a
        index = i+1
print(max_n)
print(index)
```
풀이 : 배열을 쓰지 않고 반복문으로 그냥 처리하는 게 가장 효율적인 것 같아서 쓰지 않았다.
- 배열을 써서 하는 경우, 한 줄씩 입력이 들어오면 리스트 L에 하나씩 담은 후에, max(L)을 출력한 다음, list.index(max(list)) + 1을 출력한다. 
- **주의** : list.index() 사용 시, 원소가 중복될 경우 처음 등장하는 원소 index만 출력해준다.

## 4-3. 3052번, 나머지
[prob 3052](https://www.acmicpc.net/problem/3052) : 수 10개를 입력받은 뒤, 이를 42로 나눈 나머지를 구한다. 그 다음 서로 다른 값이 몇 개 있는지 출력한다.  
```python
cnt = []
for i in range(10):
    cnt.append(int(input()) % 42)
print(len(set(cnt)))    # use set to avoid duplicated value
```
풀이 : 숫자를 입력받고 나머지를 구할 때마다 나머지 값이 이미 있는지 확인하면서 카운트하는 것보다, 나머지를 모두 구해놓고 중복을 제거하는 편이 편하다. 원소의 중복을 제거할 때는 set()으로 집합으로 만들어버리면 쉽게 처리 가능하다.

## 4-4. 1546번, 평균
[prob 1546](https://www.acmicpc.net/problem/1546) : 시험 점수 중 최댓값을 M이라고 할 때, 모든 과목 점수를 점수/M*100으로 고쳤다. 고친 후의 새로운 평균을 구한다.
```python
n = int(input())
li = list(map(int, input().split()))
max_n = max(li)
score_sum = 0
for i in li:
    score_sum += i / max_n * 100
print(score_sum/n)
```
풀이 : 문제 조건을 차례대로 따라가면 된다. for 사용 시, i에 range()를 써서 index를 담기보다 list 원소를 그대로 담는 게 편하다.

## 4-5. 8958번, OX 퀴즈
[prob 8958](https://www.acmicpc.net/problem/8958) : "OOXXOXXOOO"와 같은 OX퀴즈의 결과가 있다. 문제를 맞은 경우 그 문제의 점수는 그 문제까지 연속된 O의 개수가 된다. 예를 들어, 10번 문제의 점수는 3이 된다.

"OOXXOXXOOO"의 점수는 1+2+0+0+1+0+0+1+2+3 = 10점이다.

OX퀴즈의 결과가 주어졌을 때, 점수를 구한다.
```python
n = int(input())    # test case
for i in range(n):
    score_sum, cnt = 0, 0
    ox = input()
    for i in list(ox):
        if i == 'O':
            cnt += 1
            score_sum += cnt
        elif i == 'X':
            cnt = 0
    print(score_sum)
```
풀이 : list(string)의 경우, string의 문자들이 분리되어 리스트가 구성된다.

## 4-6. 4344번, 평균은 넘겠지
[prob 4344](https://www.acmicpc.net/problem/4344) : 각 케이스마다 평균을 넘는 학생들의 비율을 반올림하여 소수점 셋째자리까지 출력한다.
```python
n = int(input())
for i in range(n):
    li = list(map(int, input().split()))
    avg = (sum(li[1:])) / li[0]
    cnt = 0
    for j in li[1:]:
        if avg < j:
            cnt += 1
    rate = cnt / li[0] * 100
    print(f'{rate:.3f}%')
```
풀이 : list[a:b]로 list안의 a인덱스부터 b-1까지 잘라서 리스트를 구성한다. 소수점 자리를 정할 때는 f-string 사용 시, 중괄호 { } 안에서 : 구분자 이용 후 .자릿수f를 써준다. f는 float을 의미한다.

# step4 : 1차원 배열 요약
list를 slice하는 방법, list 파이썬 내장 함수 (list.index() 등), iterable 매개변수로 가지는 함수 (max(), min() 등) 사용, set()으로 중복 제거 등을 알고 있으면 좋을 것 같다.