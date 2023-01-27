---
title: "Baekjoon problem step14 num theory and combinatorics (정수론 및 조합론)"
excerpt: "Step14 정수론 및 조합론 풀이집입니다."
date: 2022-11-08
categories:
    - Algorithm
tags:
    - Baekjoon
last_modified_at: 2022-11-08
---
- Step14 정수론 및 조합론 풀이집입니다.

# num theory and combinatorics, 정수론 및 조합론
정수론 및 조합론 - [step 14 문제 리스트](https://www.acmicpc.net/step/14)  
정수론 및 조합론을 배웁니다. 

## 14-1. 5086번, 배수와 약수
[prob 5086](https://www.acmicpc.net/problem/5086) : 각 테스트 케이스마다 첫 번째 숫자가 두 번째 숫자의 약수라면 factor를, 배수라면 multiple을, 둘 다 아니라면 neither를 출력한다.
```python
# 배수와 약수
# 1 : first one is a factor of second,
# 2 : first one is a multiple of second
# 3 : first one is neither a factor nor a multiple of second
import sys
input = sys.stdin.readline

while True:
    a, b = map(int, input().split())
    if a == 0 and b == 0:
        break
    if b % a == 0:
        print("factor")
    elif a % b == 0:
        print("multiple")
    else:
        print("neither")
```
풀이 : 각 수가 서로 나눠 떨어지는지 확인한다.

## 14-2. 1037번, 약수
[prob 1037](https://www.acmicpc.net/problem/1037) : 양수 A가 N의 진짜 약수가 되려면, N이 A의 배수이고, A가 1과 N이 아니어야 한다. 어떤 수 N의 진짜 약수가 모두 주어질 때, N을 구하는 프로그램을 작성하시오
```python
# 약수
# 가장 큰 진약수 * 가장 작은 진약수 = 원래 수
import sys
input = sys.stdin.readline

n = int(input())
nums = list(map(int, input().split()))
print(min(nums) * max(nums))
```
풀이 : 가장 큰 진약수 * 가장 작은 진약수 = 원래 수 임을 알아야 풀 수 있다.

## 14-3. 2609번, 최대공약수와 최소공배수
[prob 2609](https://www.acmicpc.net/problem/2609) : 두 개의 자연수를 입력받아 최대 공약수와 최소 공배수를 출력하는 프로그램을 작성하시오.
```python
# 최대공약수와 최대공배수
# 최대공약수와 최대공배수
a, b = map(int, input().split())
cd = 1
while True:
    for i in range(2, min(a, b) + 1):
        if a % i == 0 and b % i == 0:
            cd *= i
            a //= i
            b //= i
            break
    else:
        break
print(cd)
print(cd * a * b)
```
풀이 1 : a와 b를 공통으로 나눠 떨어질 때까지 나눠서 최대공약수를 구하고, 최대공약수에 나눠진 a와 b를 곱해서 최소공배수를 만든다.

```python
# 최대공약수와 최대공배수
a, b = map(int, input().split())

def gcd(a, b):
    while b > 0:
        a, b = b, a % b
    return a

def lcm(a, b):
    return a * b // gcd(a, b)

print(gcd(a, b))
print(lcm(a, b))
```
풀이 2 : Euclidean algorithm, 유클리드 호제법을 사용한 풀이 방법이다. 
- 유클리드 호제법 : 두 자연수 a, b (a > b)에 대해 a를 b로 나눈 나머지를 r이라 할 때, a와 b의 최대공약수는 b와 r의 최대공약수와 같다. 이를 나머지가 0이 될 때까지 반복하면 a와 b의 최대공약수를 구할 수 있다.
- 최소공배수는 a, b의 곱을 a, b의 최대 공약수로 나누면 된다. (gcd * lcm = a * b)

## 14-4. 1934번, 최소공배수
[prob 1934](https://www.acmicpc.net/problem/1934) : 테스트 케이스 T개의 A, B 쌍에 대해 최소공배수를 출력한다.
```python
# 최소공배수
import sys
input = sys.stdin.readline

def gcd(a, b):
    while b > 0:
        a, b = b, a % b
    return a

t = int(input())
for i in range(t):
    a, b = map(int, input().split())
    print(a * b // gcd(a, b))
```
풀이 : 유클리드 호제법이 반복문으로 lcm을 구하는 것보다 훨씬 빠르기 때문에 유클리드 호제법으로 구한다.

## 14-5. 2981번, 검문
[prob 2981](https://www.acmicpc.net/problem/2981) : N개의 수들이 주어지고 임의의 수 M으로 나눌 때, 나머지가 모두 같게 되는 M을 모두 찾는다. (M들을 증가하는 순서로 출력한다.)
```python
# 검문
# A = a * m + r, B = b * m + r, C = c * m + r
# A - B = (a - b)*m, B - C = (b - c)*m
# m is gcd of (A-B), (B-C)
import sys
import math
input = sys.stdin.readline

n = int(input())
nums = []
m = set()

gcd = 0
for i in range(n):
    nums.append(int(input()))
    if i == 1:
        gcd = abs(nums[1] - nums[0])
    gcd = math.gcd(abs(nums[i] - nums[i - 1]), gcd)
for i in range(2, int(gcd ** 0.5) + 1):
    if gcd % i == 0:
        m.add(i)
        m.add(gcd // i)
m.add(gcd)
m = sorted(list(m))
print(*m)
```
풀이 : A, B, C의 나머지가 같은 m을 구하기 위해 A-B, B-C의 최대공약수를 찾고 1이 아닌 약수를 순서대로 출력하면 된다. math 모듈의 gcd 함수를 쓰면 최대공약수를 쉽게 찾을 수 있다.
- 어떤 수 a의 약수를 찾을 때, 시간을 줄이기 위해, root a 까지만 탐색한다. 약수 하나를 발견하면 대응되는 다른 약수를 곱해야 a가 나오기 때문이다.

## 14-6. 3036번, 링
[prob 3036](https://www.acmicpc.net/problem/3036) : 링 N개의 반지름이 주어진다. 1번째 링이 한 바퀴 돌 때, 나머지 링은 몇 바퀴 도는지 기약 분수 형태 A/B로 출력한다.
```python
# 링
# 2*pi*r = circumference(원주)
import sys
input = sys.stdin.readline
from math import gcd

n = int(input())
rings_r = list(map(int, input().split()))
for i in range(1, n):
    gcd_n = gcd(rings_r[0], rings_r[i])
    print(f'{rings_r[0]//gcd_n}/{rings_r[i]//gcd_n}')
```
풀이 : 원주는 반지름에 비례한다. 따라서 첫째 링의 반지름을 구하고자 하는 링의 반지름으로 나누면 된다. 기약 분수 형태로 나타내기 위해, 최대공약수를 구해 나눠준 후 출력한다.

## 14-7. 11050번, 이항 계수 1
[prob 11050](https://www.acmicpc.net/problem/11050) : 자연수 N과 정수 K가 주어질 때 이항계수 nCk를 구하는 프로그램을 작성한다.
```python
# 이항 계수 1
from math import factorial

n, k = map(int, input().split())
print(factorial(n) // (factorial(n - k) * factorial(k)))
```
풀이 : nCk = n! / (k! * (n-k)!)임을 math 라이브러리의 factorial 함수를 활용하여 나타내준다.

## 14-8. 11051번, 이항 계수 2
[prob 11051](https://www.acmicpc.net/problem/11051) : 자연수 N과 정수 K가 주어질 때, nCk를 10007로 나눈 나머지를 출력한다.
```python
# 이항 계수 2 using DP
# Pascal's triangle
n, k = map(int, input().split())
pascal = [[1], [1, 1]]
for i in range(3, n + 1):
    pascal.append([1] * i)
for i in range(2, n):
    for j in range(1, i):
        pascal[i][j] = pascal[i-1][j-1] + pascal[i-1][j]
if k == 0 or k == n:
    print(1)
else:
    print((pascal[n-1][k-1] + pascal[n-1][k]) % 10007)
```
풀이 1 : Pascal's triangle을 활용하여 DP로 구하는 방법이다. 
- 파스칼의 삼각형 nCr + nCr+1 = n+1Cr+1

```python
# 이항 계수 2
from math import factorial
n, k = map(int, input().split())
nCk = factorial(n) // (factorial(n-k) * factorial(k))
print(nCk % 10007)
```
풀이 2 : 위의 문제처럼 factorial 함수를 활용하여 이항계수를 구하고 10007로 나눈 나머지를 구한다.

## 14-9. 1010번, 다리 놓기
[prob 1010](https://www.acmicpc.net/problem/1010) : 강 왼쪽에 N개의 후보지, 오른쪽에 M개의 후보지가 존재한다. (N <= M) 다리끼리 서로 겹쳐질 수 없을 때, 최대로 다리를 지을 수 있는 경우의 수를 출력한다.
```python
# 다리 놓기
# mCn
import sys
from math import factorial
input = sys.stdin.readline

t = int(input())
for i in range(t):
    n, m = map(int, input().split())
    print(factorial(m) // (factorial(m-n) * factorial(n)))
```
풀이 : M개의 후보지에서 N개의 후보지를 선택하는 경우의 수와 같다.

## 14-10. 9375번, 패션왕 신혜빈
[prob 9375](https://www.acmicpc.net/problem/9375) : n개의 의상 이름, 의상 종류들이 주어질 때, 다른 조합으로 옷을 입는 경우의 수를 구한다. 
```python
# 패션왕 신해빈
import sys
input = sys.stdin.readline

t = int(input())

for case in range(t):
    n = int(input())
    dic = {}
    for i in range(n):
        cloth, category = input().split()
        if category not in dic:
            dic[category] = [cloth]
        else:
            dic[category].append(cloth)

    cnt = [len(dic[i]) for i in dic]
    result = 1
    for i in cnt:
        result *= (i + 1)
    print(result - 1)
```
풀이 : 각 카테고리별로 골라서 조합하기 때문에 안고르는 경우를 +1 해서 각 카테고리 별로 개수를 다 곱해준다음 모두 안고르는 경우 -1을 해주면 된다.

## 14-11. 1676번, 팩토리얼 0의 개수
[prob 1676](https://www.acmicpc.net/problem/1676) : n!에서 뒤에서부터 처음 0이 아닌 숫자가 나올 때까지 0의 개수를 구하는 프로그램을 출력한다.
```python
# 팩토리얼 0의 개수
n = int(input())
cnt = 0
while n >= 5:
    cnt += n // 5
    n //= 5
print(cnt)
```
풀이 : n!의 뒷자리 0의 개수를 알기 위해서는 5로 나눴을 때 몫만큼 count를 해주면된다. factorial 이기 때문에 2의 개수가 5의 출현보다 많기 때문에 고려하지 않아도 된다. 하지만, 25, 125 등 5의 제곱수들은 5를 여러개 포함하고 있기 때문에 n이 5보다 큰 경우동안, n을 5로 나눠주면 update하면서 count를 계속한다.

## 14-12. 2004번, 조합 0의 개수
[prob 2004](https://www.acmicpc.net/problem/2004) : nCm의 끝자리 0의 개수를 출력한다.
```python
# 조합 0의 개수
# nCm = n! / (n-m)!(m!)
# 2, 5 counting

def n_cnt(n, num):
    cnt = 0
    while num >= n:
        num //= n
        cnt += num
    return cnt

n, m = map(int, input().split())
two_cnt = n_cnt(2, n) - n_cnt(2, n-m) - n_cnt(2, m)
five_cnt = n_cnt(5, n) - n_cnt(5, n-m) - n_cnt(5, m)
print(min(two_cnt, five_cnt))
```
풀이 : 뒤에 붙는 0의 개수를 세려면 2와 5 개수 중 min값을 찾으면 된다. 이항계수는 factorial의 곱으로 이뤄져있기 때문에, factorial에서 count 함수를 만들어 개수를 구해준다.

# step14 : 정수론 및 조합론 요약
num theory and combinatorics에서는 배수, 약수, 최대공약수, 최소공배수, 이항 계수, factorial 등을 다뤘다. math 모듈의 factorial, gcd 함수와 유클리드 호제법을 활용하는 방법을 배웠다.