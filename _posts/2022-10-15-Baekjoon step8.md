---
title: "Baekjoon problem step8 basic math2 (기본 수학2)"
excerpt: "Step8 기본 수학2 풀이집입니다."
date: 2022-10-15
categories:
    - Algorithm
tags:
    - Baekjoon
last_modified_at: 2022-10-15
---
- Step8 기본 수학2 풀이집입니다.

# basic math2, 기본 수학2
기본 수학2 - [step 8 문제 리스트](https://www.acmicpc.net/step/10)  
기본적인 수학 개념을 코드로 표현하는 방법을 배웁니다.  

## 8-2. 1978번, 소수 찾기
[prob 1978](https://www.acmicpc.net/problem/1978) : 주어진 수 N 개 중에서 소수가 몇 개인지 찾아서 출력한다.
```python
import math
n = int(input())
a = list(map(int, input().split()))
cnt = n
for i in a:
    if i == 1:
        cnt -= 1
        continue
    for j in range(2, int(math.sqrt(i)) + 1):
        if j != i and i % j == 0:
            cnt -= 1
            break
print(cnt)
```
풀이 : 어떤 정수 a가 소수인지 확인하는 방법으로 2부터 a까지 키워가며 나눴을 때 나누어 떨어지는 지 확인할 수 있다. 하지만 이것보다 a가 소수가 아니려면 a = b * c의 형태를 가져야 함을 이용하면 좋다. b나 c 둘은 같거나 한쪽이 작기 때문에 루트 a까지만 확인하면 더 효울적이다.

## 8-2. 2581번, 소수
[prob 2581](https://www.acmicpc.net/problem/2581) : 자연수 M과 N이 주어질 때 M이상 N이하의 자연수 중 소수인 것을 모두 골라 이들 소수의 합과 최솟값을 찾는 프로그램을 작성하시오.

```python
import math
m = int(input())
n = int(input())
a = []
for i in range(m, n + 1):
    if i == 1:
        continue
    else:
        a.append(i)
    for j in range(2, int(math.sqrt(i)) + 1):
        if j != i and i % j == 0:
            a.remove(i)
            break
if not a:
    print(-1)
else:
    print(sum(a))
    print(min(a))
```
풀이 : 위의 문제와 마찬가지로 소수인지 확인하여 리스트에 담을지 말지 확인한다. 
- not 논리연산자는 논리조건을 반대로 바꾼다.
- 파이썬에서 False로 취급하는 것들은 다음과 같다. : None, False, 0인 숫자들 (0, 0.0, 0j(허수)), 빈 리스트, 빈 문자열, 빈 튜플, 빈 셋 등이 있다.
- 따라서 위의 코드에서 a가 빈 리스트일 경우 print(-1)을 실행한다.

## 8-3. 11653번, 소인수분해
[prob 11653](https://www.acmicpc.net/problem/11653) : 정수 N이 주어졌을 때, 소인수분해하는 프로그램을 작성하시오.

```python
import math
n = int(input())
if n == 1:
    exit()
else:
    while n > 1:
        for i in range(2, n + 1):
            if n % i == 0:
                print(i)
                n = n // i
                break
```
풀이 : 인수를 작은 것부터 출력하면 되기 때문에 2부터 n까지 인수 확인, 출력해주고 n을 인수로 나눠가며 진행하면 된다.
- exit() 코드로 프로그램을 종료시킬 수 있다.

## 8-4. 1929번, 소수 구하기
[prob 1929](https://www.acmicpc.net/problem/1929) : 달팽이는 높이가 V미터인 나무 막대를 올라갈 것이다. 달팽이는 낮에 A미터 올라갈 수 있다. 하지만, 밤에 잠을 자는 동안 B미터 미끄러진다. 또, 정상에 올라간 후에는 미끄러지지 않는다.

달팽이가 나무 막대를 모두 올라가려면, 며칠이 걸리는지 구하는 프로그램을 작성하시오.
  
```python
from math import ceil
a, b, v = map(int, input().split())
day = ceil((v - a) / (a - b))
print(day + 2)
```
풀이 : 정상에 올라가기 전까지는 매일 a - b만큼 올라간다. 정상에 도달한 후에는 b만큼 미끄러질 필요가 없기 때문에 이를 고려해야 한다.  

따라서 V - A <= (A - B) * day를 만족하는 day를 찾고 day + 2을 구하면 정답니다. 다시 정리하면 (V - A) / (A - B) <= day를 만족하는 day를 찾으면 된다. day는 정수 이므로 올림 ceil()을 취해준다.

## 8-5. 4948번, 베르트랑 공준
[prob 4948](https://www.acmicpc.net/problem/4948) : 베르트랑 공준은 임의의 자연수 n에 대하여, n보다 크고, 2n보다 작거나 같은 소수는 적어도 하나 존재한다는 내용을 담고 있다. 이 명제는 조제프 베르트랑이 1845년에 추측했고, 파프누티 체비쇼프가 1850년에 증명했다. 자연수 n이 주어졌을 때, n보다 크고, 2n보다 작거나 같은 소수의 개수를 구하는 프로그램을 작성하시오. 
```python
from math import sqrt

def prime(n):
    if n == 1:
        return False
    elif n == 2:
        return True
    else:
        for i in range(2, int(sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True

check_list = list(range(2, 2 * 123456 + 1)) # 123456 : max n in problem
prime_list = []
for i in check_list:
    if prime(i):
        prime_list.append(i)

n = int(input())
while n != 0:
    cnt = 0
    for i in prime_list:
        if n < i <= 2 * n:
            cnt += 1
    print(cnt)
    n = int(input())
```
풀이 : 테스트 케이스 별로 n부터 2n까지 정수에 대하여 소수인지 아닌지 일일이 확인하기보다, 문제에서 나올수 있는 n에 대하여 소수 리스트를 만들어 둔 후, 각 케이스에 대하여 소수 리스트에서 카운트하는 것이 케이스가 많아질수록 훨씬 빠르다.

## 8-6. 9020번, 골드바흐의 추측
[prob 9020](https://www.acmicpc.net/problem/9020) : 골드바흐의 추측은 유명한 정수론의 미해결 문제로, 2보다 큰 모든 짝수는 두 소수의 합으로 나타낼 수 있다는 것이다. 이러한 수를 골드바흐 수라고 한다. 또, 짝수를 두 소수의 합으로 나타내는 표현을 그 수의 골드바흐 파티션이라고 한다. 10000보다 작거나 같은 모든 짝수 n에 대한 골드바흐 파티션은 존재한다.

2보다 큰 짝수 n이 주어졌을 때, n의 골드바흐 파티션을 출력하는 프로그램을 작성하시오. 만약 가능한 n의 골드바흐 파티션이 여러 가지인 경우에는 두 소수의 차이가 가장 작은 것을 출력한다.
```python
check_list = [False, False] + [True] * 10000
for i in range(2, 101): # Sieve of Eratosthenes
    if check_list[i] == True:
        for j in range(2 * i, 10001, i):
            check_list[j] = False
t = int(input())
for i in range(t):
    n = int(input())
    L = n // 2
    R = L
    for _ in range(10000):
        if check_list[L] and check_list[R]:
            print(L, R)
            break
        L -= 1
        R += 1
```
풀이 : Sieve of Eratosthenes, 에라토스테네스의 체를 이용하여 0부터 10000까지 소수들을 찾고, 짝수 n에 대하여 n / 2로 나누어 R, L을 만들고 각각 1씩 키우고 낮추며 둘다 소수가 될 때까지 진행한다.
- Sieve of Eratosthenes 에라토스테네스의 체 : 어떤 범위 내에서 소수를 모두 체크할 때, 소수 본인을 제외한 소수의 배수들은 소수가 아니므로 제거해가는 방법이다. 2부터 수를 키워가며 배수들을 체크해간다.

# step8 : 기본 수학2 요약
basic math2에서는 수학적 문제 상황을 해결하는 방법을 배웠다. 그 과정에서 소수, 소인수 분해 등에 대해 주로 다뤘다. 어떤 수 n에 대해 소수인지 확인하는 방법으로 2부터 루트 n까지 키우가며 나눠떨어지는 게 있는지 확인하면 되고, 어떤 범위 내에서 소수를 모두 체크하려면 에라토스테네스의 체를 활용하면 된다.