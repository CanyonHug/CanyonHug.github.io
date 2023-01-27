---
title: "Baekjoon problem step13 geometry1 (기하 1)"
excerpt: "Step13 기하 1 풀이집입니다."
date: 2022-11-08
categories:
    - Algorithm
tags:
    - Baekjoon
last_modified_at: 2022-11-08
---
- Step13 기하 1 풀이집입니다.

# geometry1, 기하 1
기하 1 - [step 13 문제 리스트](https://www.acmicpc.net/step/13)  
다양한 기하 문제를 코드로 나타내고 해결하는 방법을 배웁니다..  

## 13-1. 1085번, 직사각형에서 탈출
[prob 1085](https://www.acmicpc.net/problem/1085) : (x, y)에 존재할 때, 직사각형은 각 변이 좌표축에 평행하고 왼쪽 아래 꼭짓점이 (0, 0), 오른쪽 위 꼭짓점이 (w, h)에 존재한다. 직사각형 경계선까지 가는 거리 최솟값을 출력한다. (w, h는 각각 x, y보다 크다.)
```python
# 직사각형에서 탈출
import sys
x, y, w, h = map(int, sys.stdin.readline().split())
print(min(x, y, w - x, h - y))
```
풀이 : x, y가 직사각형 안에 존재하므로 직사각형 꼭짓점으로 4가지 방향으로 그을 수 있다. 이 4가지 경우 중 min 값을 찾아서 출력한다.

## 13-2. 3009번, 네 번째 점
[prob 3009](https://www.acmicpc.net/problem/3009) : 세 점이 주어졌을 때, 축에 평행한 직사각형을 만들기 위해 필요한 네 번째 점을 출력한다.
```python
# 네 번째 점
x_li = []
y_li = []
x_4 = 0
y_4 = 0

for i in range(3):
    x, y = map(int, input().split())
    x_li.append(x)
    y_li.append(y)

for i in range(3):
    if x_li.count(x_li[i]) == 1:
        x_4 = x_li[i]
    if y_li.count(y_li[i]) == 1:
        y_4 = y_li[i]

print(x_4, y_4)
    
```
풀이 : 3 점 중 x좌표 y좌표 수를 count해서 1개 씩 있는  x좌표, y좌표가 4번째 점이다.

## 13-3. 4153번, 직각삼각형
[prob 4153](https://www.acmicpc.net/problem/4153) : 세 변의 길이가 주어질 때, 삼각형이 직각인지 확인한다.
```python
# 직각삼각형
import sys
in_put = sys.stdin.readline

while True:
    nums = list(map(int, in_put().split()))
    max_n = max(nums)
    if sum(nums) == 0:
        break
    nums.remove(max_n)
    if nums[0] ** 2 + nums[1] ** 2 == max_n ** 2:
        print("right")
    else:
        print("wrong")

```
풀이 : 피타고라스의 정리를 이용해서 풀어준다.

## 13-4. 2477번, 참외밭
[prob 2477](https://www.acmicpc.net/problem/2477) : 참외밭이 ㄱ 자를 0, 90, 180, 270도 회전한 모양 중 하나인 육각형이다. 1m^2 당 자라는 참외의 개수가 주어질 때, 방향과 길이가 주어진다. 주어진 밭에서 자라는 참외수를 출력한다. (변의 방향에서 1:동, 2:서, 3:남, 4:북)
```python
# 참외밭
import sys
input = sys.stdin.readline

k = int(input())
arr = []
big_w, big_h, big_w_idx, big_h_idx = 0, 0, 0, 0
# big rectangle - small rectangle
for i in range(6):
    arr.append(list(map(int, input().split())))
    if (arr[i][0] == 1 or arr[i][0] == 2) and big_w < arr[i][1]:
        big_w = arr[i][1]
        big_w_idx = i
    elif (arr[i][0] == 3 or arr[i][0] == 4) and big_h < arr[i][1]:
        big_h = arr[i][1]
        big_h_idx = i
# 1 : right, 2 : left, 3 : down 4 : up
# small rectangle width = difference btw 2 horizontal lines adjacent to largest height
# small rectangle height = difference btw 2 vertical lines adjacent to largest width
small_w, small_h = 0, 0
small_w = abs(arr[big_h_idx - 1][1] - arr[(big_h_idx + 1) % 6][1])
small_h = abs(arr[big_w_idx - 1][1] - arr[(big_w_idx + 1) % 6][1])
print(k * (big_w * big_h - small_w * small_h))
```
풀이 : 밭 모양이 ㄱ자를 회전시킨 모양이므로, 큰 직사각형에서 작은 직사각형을 빼서 넓이를 구해준다.

## 13-5. 3053번, 택시 기하학
[prob 3053](https://www.acmicpc.net/problem/3053) : R이 주어질 때, 유클리드 기하학에서 반지름 R인 원의 넓이, 택시 기하학에서 반지름이 R인 원의 넓이를 출력한다. (오차는 0.0001까지 허용, 소수점 6자리까지 표시)
```python
# 택시 기하학
import sys
from math import pi
input = sys.stdin.readline

r = int(input())
print(round(pi * r ** 2, 6))
print(round(2 * r ** 2, 6))
```
풀이 : 택시기하학에서 원은 정사각형이다.

## 13-6. 1002번, 터렛
[prob 1002](https://www.acmicpc.net/problem/1002) : a의 좌표 (x1, y1)과 b의 좌표 (x2, y2)가 주어졌다. a에서 c까지의 거리 r1과 b에서 c까지의 거리 r2가 주어질 때, c가 있을 수 있는 좌표의 수를 출력한다.
```python
# 터렛
import sys
input = sys.stdin.readline

t = int(input())
for i in range(t):
    x1, y1, r1, x2, y2, r2 = map(int, input().split())
    dist = ((x1 - x2)**2 + (y1 - y2)**2)**0.5
    if dist == 0 and r1 == r2:
        print(-1)
    elif dist == r1 + r2 or dist == abs(r1 -r2):
        print(1)
    elif abs(r1 - r2) < dist < r1 + r2:
        print(2)
    else:
        print(0)
```
풀이 : 중심이 x1, y1이고 반지름이 r1인 원과 중심이 x2, y2이고 반지름이 r2인 원이 만날 수 있는 case를 구한다.  
    1. 두 원이 완전히 겹치는 경우
    2. 한 점에서 접하는 경우
    3. 두 점에서 만나는 경우
    4. 만나지 않는 경우

## 13-7. 1004번, 어린 왕자
[prob 1004](https://www.acmicpc.net/problem/1004) : 출발점과 도착점, n개의 행성의 반지름과 중점이 주어질 때, 행성 진입/이탈 횟수를 최소화하여 여행하고자 한다. 이때 최소 진입/이탈 횟수를 출력한다.
```python
# 어린 왕자
import sys
from math import dist
input = sys.stdin.readline

t = int(input())
for i in range(t):
    x1, y1, x2, y2 = map(int, input().split())
    # count the case when circle include only one of start pt, end pt
    n = int(input())
    cnt = 0
    for i in range(n):
        x, y, r = map(int, input().split())
        if (dist((x1, y1), (x, y)) < r) ^ (dist((x2, y2), (x, y)) < r):
            cnt += 1
    print(cnt)
```
풀이 : 출발점과 도착점 중 하나만 포함하는 경우만 진입, 이탈 횟수로 count된다.
- ^ 연산자는 XOR연산자 (exclusive OR)로 2 개 중 하나만 참인 경우 참이다.

# step13 : 기하 1 요약
geometry1에서는 기하 문제를 해결하는 방법을 배운다. 수학 관련 라이브러리를 사용할 경우 math 모듈을 import하여 사용할 수 있다. 기하 상황을 이해한 다음 논리연산과 조건문으로 case를 나누고 처리한다.