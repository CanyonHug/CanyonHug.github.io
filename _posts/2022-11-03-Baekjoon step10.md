---
title: "Baekjoon problem step10 recursive (재귀)"
excerpt: "Step10 재귀 풀이집입니다."
date: 2022-11-03
categories:
    - Algorithm
tags:
    - Baekjoon
last_modified_at: 2022-11-03
---
- Step10 재귀 풀이집입니다.

# recursive, 재귀
재귀 - [step 10 문제 리스트](https://www.acmicpc.net/step/10)  
재귀함수를 활용하는 방법을 배웁니다. 재귀함수란 정의 단계에서 자신을 재참조하는 함수를 뜻한다.

## 10-1. 10872번, 팩토리얼
[prob 10872](https://www.acmicpc.net/problem/10872) : N이 주어질 때, N!을 출력한다.
```python
# 팩토리얼
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)

n = int(input())
print(factorial(n))
```
풀이 : factorial 함수를 정의할 때, return 값에 parameter 값을 하나 빼 준 factorial 함수를 포함하여 구성해준다.

## 10-2. 10870번, 피보나치 수 5
[prob 10870](https://www.acmicpc.net/problem/10870) : n이 주어질 때, n번째 피보나치 수를 출력한다. (피보나치 수는 Fn = Fn-1 + Fn-2 이다.)
```python
# 피보나치 수 5
def Fibo(n):
    if n <= 1:
        return n
    return Fibo(n - 1) + Fibo (n - 2)

n = int(input())
print(Fibo(n))
```
풀이 : 피보나치 수 정의 자체부터 재귀적으로 정의됐기 때문에 쉽게 작성 가능하다.
- 재귀함수를 사용할 경우, 이미 계산했던 작은 문제들을 여러 번 반복해서 계산해 비효율적일 수 있다. 이때, Dinamic Programming (동적 계획법)을 사용하면 효율적으로 풀 수 있다. memorization이 추가되어 반복 계산 없이 효율적으로 가능하다.
- 재귀함수에 대한 자세한 설명은 다음에서 확인 가능하다. <https://hongjw1938.tistory.com/47>

## 10-3. 25501번, 재귀의 귀재
[prob 25501](https://www.acmicpc.net/problem/25501) : Palindrome은 앞에서 읽을 때와 뒤에서 읽을 때가 같은 문자열을 말한다. 주어진 문자열이 Palindrome인지, 판별과정에서 recursion 함수를 몇 번 호출하는지 출력한다.
```python
import sys
input = sys.stdin.readline

def recursion(s, l, r):
    if(l >= r):
        return 1, l + 1
    elif (s[l] != s[r]):
        return 0, l + 1
    else:
        return recursion(s, l + 1, r - 1)

def isPalindrome(s):
    return recursion(s, 0, len(s) - 1)

T = int(input())
for i in range(T):
    s = input().strip()
    print(*isPalindrome(s))
```
풀이 : 문제에 알고리즘을 제시해줬기 때문에 그대로 따라가면 된다. recursion 횟수는 l + 1만큼 반복된다.
- 파이썬에서 함수 반환 값을 여러 개 반환할 때는 tuple 형태이기 때문에 print(*tuple)을 사용하면 괄호, 콤마 없이 출력가능하다.

## 10-4. 24060번, 알고리즘 수업 - 병합 정렬1
[prob 24060](https://www.acmicpc.net/problem/24060) : 배열 A를 정렬했을 때, k번째 저장되는 수를 출력한다. (저장 횟수가 k보다 작으면 -1을 출력한다.)
```python
import sys
input = sys.stdin.readline

def merge(arr):
    if len(arr) <= 1:
        return arr
    mid = (len(arr) + 1) // 2
    # left side is bigger when the array is odd. So plus one before dividing.
    L = merge(arr[:mid])
    R = merge(arr[mid:])

    i, j = 0, 0
    arr_sorted = []
    while i < len(L) and j < len(R):
        if L[i] < R[j]:
            arr_sorted.append(L[i])
            sort_sequence.append(L[i])
            i += 1
        else:
            arr_sorted.append(R[j])
            sort_sequence.append(R[j])
            j += 1
    while i < len(L):
        arr_sorted.append(L[i])
        sort_sequence.append(L[i])
        i += 1
    while j < len(R):
        arr_sorted.append(R[j])
        sort_sequence.append(R[j])
        j += 1
    
    return arr_sorted

n, k = map(int, input().split())
A = list(map(int, input().split()))
sort_sequence = []
A = merge(A)

if len(sort_sequence) < k:
    print(-1)
else:
    print(sort_sequence[k - 1])
```
풀이 : 정렬 순서를 기록하는 다른 리스트를 만들어 merge sort의 merge 과정 중에 계속 기록해준다.

## 10-5. 2447번, 별 찍기 -10
[prob 2447](https://www.acmicpc.net/problem/2447) : 3의 거듭제곱 N이 주어질 때, N*N 정사각형 모양의 가운데가 비어있는 패턴으로 별을 찍어서 출력한다.
```python
# 별 찍기 - 10
def Star(n):
    if n == 1:
        return ['*']
    arr = Star(n // 3)
    star_list = []

    for i in arr:
        star_list.append(i * 3)
    for i in arr:
        star_list.append(i + ' ' * (n//3) + i)
    for i in arr:
        star_list.append(i * 3)
    
    return star_list

n = int(input())
print('\n'.join(Star(n)))
```
풀이 : 1, 3, 9 크기로 모양을 키워가며 규칙을 찾는다. 규칙을 n으로 표현하여 구하고 리스트에 담아서 출력한다.

## 10-6. 11729번, 하노이 탑 이동 순서
[prob 11729](https://www.acmicpc.net/problem/11729) : 원판의 개수 N이 주어질 때, 하노이 탑의 최소 이동 횟수를 출력한다.
```python
# 하노이 탑 이동 순서
def hanoi(n, start, end):
    if n == 1:
        print(start, end)
        return
    hanoi(n - 1, start, 1+2+3 - start - end) # n-1 block move
    print(start, end) # lowest disk moved
    hanoi(n-1, 1+2+3 - start - end, end) # n-1 block moved on lowest disk

n = int(input())
print(2**n - 1)
hanoi(n, 1, 3)
```
풀이 : 규칙을 찾기가 어려워서 사이트를 참고했다. : <https://study-all-night.tistory.com/6>

# step10 : 재귀 요약
recursive에서는 재귀함수를 다루는 방법을 배웠다. 재귀 함수는 함수 정의 부분에서 호출 시 자기 자신을 호출하거나 중간 연산 시 자기 자신을 호출하는 함수를 뜻한다. 재귀 함수를 사용하는 경우, 큰 부분과 작은 부분이 유사한 구조를 가지기 때문에 규칙을 파악하는 것이 중요하다.