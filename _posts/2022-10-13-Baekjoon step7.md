---
title: "Baekjoon problem step7 basic math1 (기본 수학1)"
excerpt: "Step7 기본 수학1 풀이집입니다."
date: 2022-10-13
categories:
    - Algorithm
tags:
    - Baekjoon
last_modified_at: 2022-10-14
---
- Step7 기본 수학1 풀이집입니다.

# basic math1, 기본 수학1
기본 수학1 - [step 7 문제 리스트](https://www.acmicpc.net/step/8)  
기본적인 수학 개념을 코드로 표현하는 방법을 배웁니다.  

## 7-1. 1712번, 손익분기점
[prob 1712](https://www.acmicpc.net/problem/1712) : 고정 비용 A만원, 가변 비용 B만원, 노트북 판매 가격 C만원일 때, 총 수입 비용이 총 비용(고정비용 + 가변비용)보다 많아지는 지점(손익분기점 break-even point)를 구한다. (손익분기점 판매량을 출력, 존재하지 않으면 -1 출력)
```python
a, b, c = map(int, input().split())
s = 0
if b >= c:
    s = -1
else:
    s = int(a / (c - b)) + 1
print(s)
```
풀이 : 가변 비용이 판매 비용 이상이면 손해이므로 -1 출력, 그 외에는 총 비용 c * 판매량 > a + b * 판매량 일 때, 판매량을 구해야 하므로, 판매량 > a / (c -b)인데, 판매량은 정수이므로 int() 후 + 1을 취한다.

## 7-2. 2292번, 벌집
[prob 2292](https://www.acmicpc.net/problem/2292) : 아래 그림과 같이 육각형으로 이루어진 벌집이 있다.  
![honeycomb](/assets/images/baekjoon_2292.jpg){: width="30%" height="30%" .align-center}  
숫자 N이 주어졌을 때, 벌집의 중앙 1에서 N번 방까지 최소 개수의 방을 지나서 갈 때 몇 개의 방을 지나가는지(시작과 끝을 포함하여)를 계산하는 프로그램을 작성하시오. 예를 들면, 13까지는 3개, 58까지는 5개를 지난다.

```python
import math
n = int(input())
a = math.ceil((n - 1) / 6)
i = 1
while True:
    if a > 0:
        a -= i
        i += 1
    else:
        print(i)
        break
```
풀이 : 1을 중심으로 정육각형 테두리에 해당하는 숫자들은 모두 이동 횟수가 같다. 따라서 정육각형 테두리에 해당하는 숫자들을 모두 같게 처리해주려면 n-1 / 6에 올림을 취해주면 된다. 이때, 숫자들은 등차가 1인 등차수열만큼 커진다. (0, 1, 3, 6 등)따라서 i값을 1씩 키우면서 빼줘서 카운트를 한다. 

```python
n = int(input())
nums_pileup = 1  # 벌집의 개수, 1개부터 시작
cnt = 1
while n > nums_pileup :
    nums_pileup += 6 * cnt  # 벌집이 6의 배수로 증가
    cnt += 1  # 반복문을 반복하는 횟수
print(cnt)
```
출처 : <https://ooyoung.tistory.com/82>
- 굳이 테두리를 같은 수로 처리하지 않고,그냥 1부터 6의 배수로 키워갈 때 원래 숫자보다 커지는 순간을 알아내면 된다. 이 방식이 더 직관적인 것 같다.

## 7-3. 1193번, 분수 찾기
[prob 1193](https://www.acmicpc.net/problem/1193) : 무한히 큰 배열에 다음과 같이 분수들이 적혀있다.  
![honeycomb](/assets/images/baekjoon_2292.jpg){: width="30%" height="30%" .align-center}  

이와 같이 나열된 분수들을 1/1 → 1/2 → 2/1 → 3/1 → 2/2 → … 과 같은 지그재그 순서로 차례대로 1번, 2번, 3번, 4번, 5번, … 분수라고 하자. X가 주어졌을 때, X번째 분수를 구하는 프로그램을 작성하시오.
```python
n = int(input())
# 2, 3, 4, 5 ~ denom + nomin
sum = 0
i = 0
while n > sum:
    i += 1
    sum += i
# If i is even, counting is downward. #So recounting upward
denom = i                        
numer = 1
for j in range(sum - n):
    denom -= 1
    numer += 1
if i % 2 == 0:
    print(f'{denom}/{numer}')
else:
    print(f'{numer}/{denom}')
```
풀이 : 1/1 | 1/2, 2/1 | 3/1, 2/2, 1/3 으로 이어지는데, 분자와 분모의 합이 1씩 커지는 것을 알 수 있다. 또한 합이 a인 분수들은 a-1개만큼 존재하기 때문에, 주어진 수 n이 속한 그룹(분자, 분모의 합)까지 이동한다. 이때, i가 짝수면 분자가 작아지고 분모가 커지도록 count하고 홀수면 반대로 진행한다.

## 7-4. 2869번, 달팽이는 올라가고 싶다
[prob 2869](https://www.acmicpc.net/problem/2869) : 달팽이는 높이가 V미터인 나무 막대를 올라갈 것이다. 달팽이는 낮에 A미터 올라갈 수 있다. 하지만, 밤에 잠을 자는 동안 B미터 미끄러진다. 또, 정상에 올라간 후에는 미끄러지지 않는다.

달팽이가 나무 막대를 모두 올라가려면, 며칠이 걸리는지 구하는 프로그램을 작성하시오.
  
```python
from math import ceil
a, b, v = map(int, input().split())
day = ceil((v - a) / (a - b))
print(day + 1)
```
풀이 : 정상에 올라가기 전까지는 매일 a - b만큼 올라간다. 정상에 도달한 후에는 b만큼 미끄러질 필요가 없기 때문에 이를 고려해야 한다.  

따라서 V - A <= (A - B) * day를 만족하는 day를 찾고 day + 1을 구하면 정답니다. 다시 정리하면 (V - A) / (A - B) <= day를 만족하는 day를 찾으면 된다. day는 정수 이므로 올림 ceil()을 취해준다.

## 7-5. 10250번, ACM 호텔
[prob 10250](https://www.acmicpc.net/problem/10250) : 호텔 정문으로부터 걷는 거리가 가장 짧도록 방을 배정하는 프로그램을 작성하고자 한다.

문제를 단순화하기 위해서 호텔은 직사각형 모양이라고 가정하자. 각 층에 W 개의 방이 있는 H 층 건물이라고 가정하자 (1 ≤ H, W ≤ 99). 그리고 엘리베이터는 가장 왼쪽에 있다고 가정하자(그림 1 참고).

![honeycomb](/assets/images/baekjoon_10250.jpg){: width="50%" height="50%" .align-center}  

방 번호는 YXX 나 YYXX 형태인데 여기서 Y 나 YY 는 층 수를 나타내고 XX 는 엘리베이터에서부터 세었을 때의 번호를 나타낸다. 손님은 엘리베이터를 타고 이동하는 거리는 신경 쓰지 않는다. 다만 걷는 거리가 같을 때에는 아래층의 방을 더 선호한다. 초기에 모든 방이 비어있다고 가정하에 이 정책에 따라 N 번째로 도착한 손님에게 배정될 방 번호를 계산하는 프로그램을 작성한다.

```python
t = int(input())
for i in range(t):
    h, w, n = map(int, input().split())
    n_w = n // h    # // : 소수점 이하 버림
    n_h = n % h
    if n_h != 0:
        n_w += 1
    else:
        n_h = h
    print(f'{n_h}0{n_w}' if n_w < 10 else f'{n_h}{n_w}')
```
풀이 : 방을 채울 때, h 우선으로 채워지기 때문에, w가 n / h를 올림한 것과 같다. h도 n을 h로 나눴을 때 나머지에 따라 설정해준다. 그리고 호수의 w가 10을 넘어가는지에 따라 출력을 달리해준다.
- print()에서 조건문으로 출력을 달리하면서도 한줄에 쓰고 싶을 때 위와 같이 작성한다. print('a' if 조건 else 'b') 

## 7-6. 2775번, 부녀회장이 될테야
[prob 2775](https://www.acmicpc.net/problem/2775) : 이 아파트에 거주를 하려면 조건이 있는데, “a층의 b호에 살려면 자신의 아래(a-1)층의 1호부터 b호까지 사람들의 수의 합만큼 사람들을 데려와 살아야 한다” 는 계약 조항을 꼭 지키고 들어와야 한다.

아파트에 비어있는 집은 없고 모든 거주민들이 이 계약 조건을 지키고 왔다고 가정했을 때, 주어지는 양의 정수 k와 n에 대해 k층에 n호에는 몇 명이 살고 있는지 출력하라. 단, 아파트에는 0층부터 있고 각층에는 1호부터 있으며, 0층의 i호에는 i명이 산다.
```python
t = int(input())    # test case
for i in range(t):
    k = int(input())
    n = int(input())
    a = [[0 for j in range(n)] for i in range(k + 1)]
    for i in range(k + 1):
        for j in range(n):
            if i == 0:
                a[i][j] = j + 1
            else:
                for t in range(j + 1):
                    a[i][j] += a[i - 1][t]
    print(a[k][n - 1])
```
풀이 : n*k 짜리 2차원 배열을 만들어서 조건에 맞게 0층 k층까지 1호부터 n호까지 채워 넣는다. 2중 반복문을 사용하면 수월하게 채워넣을 수 있다.
- array = [[0 for **column** in range(10)] for **row** in range(10)] 위와 같이 2중 for 문으로 2중 리스트를 선언할 수 있다. 접근은 array[row][column]으로 할 수 있다. (코드 작성시 column이 먼저 나오니 유의해야 한다. 가로 줄을 쌓기 때문에 row가 늦게 나온다고 생각하면 좋을 것 같다.)
- **주의** : 1차원 리스트 작성시 [0] * 5와 [0 for i in range(5)]는 차이가 없지만, 2차원 리스트의 경우 [[0] * 5] * 5의 경우 얕은 복사가 일어나서 [1][1] = 1을 하면 모든 2열 값들이 1로 바뀐다. 따라서 반복문을 통해 생성해야 한다.
- 얕은 복사 (shallow copy) : 주소 값을 복사함. (같은 객체 가리킴) -> 값 변경 시 얕은 복사된 것들 같이 바뀜.
- 깊은 복사 (Deep copy) : 실제 값을 메모리 공간에 복사함.

## 7-7. 2839번, 설탕 배달
[prob 2839](https://www.acmicpc.net/problem/2839) : 3kg 봉지와 5kg 봉지만 사용해서 설탕 N kg을 배달할 때, 최소 개수의 봉지를 사용해서 배달하려고 한다. 이때 필요한 봉지의 최소 개수를 알아내야 한다.
  
```python
n = int(input())
cnt = 0
while n >= 0:
    if n % 5 == 0:
        cnt += n // 5
        print(cnt)
        break
    n -= 3
    cnt += 1
else:
    print(-1)
```
풀이1 : N kg이 5kg과 3kg으로 담아낼 수 있다면 N = a * 5 + b * 3을 만족하는 a, b가 존재해야 할 것이다. 그렇기 때문에 while문에서 N에서 3씩 덜어내면서 5로 나눠떨어지는지 확인한다. 나눠떨어지면 5kg으로 나머지를 모두 담고 개수를 출력하면 된다. 만약 3kg 5kg으로 담을 수 없다면, while else 구문을 통해 -1을 출력한다.
- while else란?  
while 조건: 부분에서, 조건이 거짓으로 판명되어 while 안의 코드들이 실행되지 않을 때, else 안의 코드들이 실행된다. 만약 break 문에 의해 반복이 끝나면 for과 마찬가지로 else 부분이 실행되지 않고 끝난다.

## 7-8. 10757번, 큰 수 A+B
[prob 10757](https://www.acmicpc.net/problem/10757) : 두 정수 A와 B를 입력받은 다음, A+B를 출력하는 프로그램을 작성하시오.

```python
a, b = map(int, input().split())
print(a + b)
```
풀이 : C에서는 데이터 타입에 따라 메모리가 일정 부분 할당되고 채워졌는데, python3에서는 정수형 데이터도 int하나로 arbitary precision을 사용하기 때문에 오버플로우에 신경쓰지 않아도 된다.
- arbitary precision : 사용할 수 있는 메모리양이 정해진 fixed-precision과 다르게 자릿수 단위로 쪼개어 배열 형태로 표현하기 때문에 오버플로우가 발생하지 않는다. 하지만, 속도가 상대적으로 느리다는 단점이 있다.  

# step7 : 기본 수학1 요약
basic math1에서는 수학적 문제 상황을 해결하는 방법을 배웠다. 그 과정에서 2차원 배열, math.ceil() 함수 등의 활용방안도 익힐 수 있었다.