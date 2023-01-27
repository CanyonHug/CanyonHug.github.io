---
title: "Baekjoon problem step11 brute-force (브루트 포스)"
excerpt: "Step11 브루트 포스 풀이집입니다."
date: 2022-11-03
categories:
    - Algorithm
tags:
    - Baekjoon
last_modified_at: 2022-11-07
---
- Step11 브루트 포스 풀이집입니다.

# brute-force, 브루트 포스
브루트 포스 - [step 11 문제 리스트](https://www.acmicpc.net/step/11)  
브루트 포스 방식을 활용하는 방법을 배웁니다. (brute force는 직역하면 무식한 힘으로, 발생할 수 있는 모든 경우를 탐색하는 완전 탐색 방식이다.)

## 11-1. 2798번, 블랙잭
[prob 2798](https://www.acmicpc.net/problem/2798) : N장의 카드가 주어질 때, M을 넘지 않으면서 M에 가장 가까운 카드 3장의 합을 출력한다.
```python
# 블랙잭
n, m = map(int, input().split())
a = list(map(int, input().split()))
sum = 0
for i in range(n):
    for j in range(i + 1, n):
        for k in range(j + 1, n):
            if  sum < a[i] + a[j] + a[k] <= m:
                sum = a[i] + a[j] + a[k]
print(sum)
```
풀이 : i, j, k를 통해 가능한 카드의 조합을 모두 탐색하면서, sum 보다 크면서 m보다 작을경우, sum을 업데이트해준다.
- 파이썬에서 부등호 조건을 여러 개 사용할 경우, and 논리연산자를 사용해도 되지만 실제 수학 부등호 작성할 때처럼 부등호를 연달아 사용할 수 있다.

## 11-2. 2231번, 분해합
[prob 2231](https://www.acmicpc.net/problem/2231) : 어떤 자연수 N에 대하여, N의 분해합은 N과 N의 각 자리수의 합을 의미한다. 어떤 자연수 M의 분해합이 N인 경우, M을 N의 생성자라고 한다. 어떤 자연수 N이 주어질 때, N의 가장 작은 생성자를 구하여 출력한다.
```python
# 분해합
n = int(input())
for i in range(1, n):
    sum = i
    temp = i
    while temp > 0:
        sum += temp % 10
        temp = temp // 10
    if sum == n:
        print(i)
        break
else:
    print(0)
```
풀이 : n보다 작은 수에 대해 분해합을 만들어보고 n이 되는 경우를 출력한다.
- 파이썬에서는 for-else 구문이 가능하다. for 문이 중간에 break 등으로 끊기지 않고 끝까지 수행될 경우, else 문이 실행된다.

## 11-3. 7568번, 덩치
[prob 7568](https://www.acmicpc.net/problem/7568) : 어떤 사람이 다른 사람보다 키, 몸무게 둘다 큰 경우 덩치가 크다고 말한다. N명의 몸무게, 키가 주어질 때, 덩치 등수를 구해서 나열된 사람의 순서대로 덩치 등수를 출력한다.
```python
# 덩치
n = int(input())
stu_li = []
for i in range(n):
    w, h = (map(int, input().split()))
    stu_li.append((w, h))

for i in stu_li:
    rank = 1
    for j in stu_li:
        if i[0] < j[0] and i[1] < j[1]:
            rank += 1
    print(rank, end = " ")
```
풀이 : 리스트를 돌면서 리스트 내의 다른 사람과 덩치 비교를 한 후, 작을 경우 rank를 올리는 방식으로 탐색한다.

## 11-4. 1018번, 체스판 다시 칠하기
[prob 1018](https://www.acmicpc.net/problem/1018) : M * N 크기의 보드가 주어질 때, 8 * 8 크기의 체스판을 만들려 한다. 임의의 위치에서 8 * 8크기로 잘라낸 후, 체스판이 되기 위해 흰색 검은색을 다시 칠해야 하는 칸의 최소 개수를 출력한다.
```python
# 체스판 다시 칠하기
m, n = map(int, input().split())
table = []
for i in range(m):
    BW_string = input()
    table.append(BW_string)
repair = []
for i in range(m + 1 - 8):
    for j in range(n + 1 - 8):
        bw_error = 0      # left top is black
        wb_error = 0      # left top is white
        for k in range(i, i + 8):
            for l in range(j, j + 8):
                if (k + l) % 2 == 0:        # even point (0, 0), (0, 2)
                    if table[k][l] != 'B':
                        bw_error += 1
                    elif table[k][l] != 'W':
                        wb_error += 1
                else:                       # odd point (0, 1), (0, 3) ~
                    if table[k][l] != 'W':
                        bw_error += 1
                    elif table[k][l] != 'B':
                        wb_error += 1
        repair.append(bw_error)
        repair.append(wb_error)
print(min(repair))
```
풀이 : 첫 이중 for 문은 8 * 8 크기로 체스판을 잘라낸 것이고, 다음 이중 for 문은 그 체스판을 돌면서 잘못 칠해진 경우를 count하는 것이다. 이때, 맨왼쪽 위가 흰색인 경우, 검은색인 경우를 나누어서 count한다.

## 11-5. 1436번, 영화감독 숌
[prob 1436](https://www.acmicpc.net/problem/1436) : 어떤 수에 적어도 3개 이상 연속으로 6이 들어가는 수를 종말의 숫자라고 한다. 어떤 수 N이 주어질 때, N번째 종말의 수를 출력한다.
```python
# 영화감독 숌
n = int(input())
cnt = 0
num = 666
while True:
    if '666' in str(num):
        cnt += 1
    if cnt == n:
        print(num)
        break
    num += 1
```
풀이 : 숫자를 다룰 때, 666이 포함되어 있는지 확인해야 되기 때문에 int가 아닌 string으로 다뤄 in을 사용해 확인을 쉽게한다. 666부터 1씩 키워가며 n번째 종말의 수를 찾아서 출력한다.

# step11 : 브루트 포스 요약
brute-force에서는 전체 경우의 수를 탐색하는 방법을 배웠다. 공간복잡도나 시간복잡도 면에서 효율적인 방법은 아니지만, 전체 경우의 수를 고려하기 때문에 유한한 case에서 답을 보장할 수 있다. 중간에 빠지는 경우가 없도록 전체를 탐색하는 것이 중요하다.