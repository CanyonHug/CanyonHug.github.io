---
title: "Baekjoon problem step3 loop (반복문)"
excerpt: "Step3 반복문 풀이집입니다."
date: 2022-10-11
categories:
    - Algorithm
tags:
    - Baekjoon
last_modified_at: 2022-10-11
---
- Step3 반복문 풀이집입니다.

# Loop, 반복문
for, while 등의 조건문 사용해보기 - [step 3 문제 리스트](https://www.acmicpc.net/step/3)  
문제 상황의 조건에 맞게 어떻게 동작할 지 배우는 단계입니다. 기초적인 logic을 통해 case가 선형적으로 풀이될 수 있도록 반복문 코드를 구성합니다.  

## 3-1. 2739번, 구구단  
[prob 2739](https://www.acmicpc.net/problem/2739) : N이 주어졌을 때, 구구단 N단을 출력한다.  
```python
a = int(input())
for i in range(1, 10):
    print(a, "*", i, "=", a * i)
```
풀이 : i에 range(start, end + 1)를 통해 순서대로 start부터 end까지 값을 대입하며 end + 1 - start 만큼 반복한다.
- **주의** : range 함수의 두번째 매개변수 값까지 반복되는 게 아니라, 두번째 매개변수 -1까지만 대입된다.

## 3-3. 10950번, A+B - 3
[prob 10950](https://www.acmicpc.net/problem/10950) : 테스트 개수 T가 주어지고, T번만큼 A, B가 주어진다. 각 케이스에 대해 A+B를 출력한다.  
```python
n = int(input())
for i in range(n):
    a, b = map(int, input().split())
    print(a + b)
```
풀이 : 간단하게 n번만큼 a + b를 시행하면 된다.

## 3-3. 8393번, 합
[prob 8393](https://www.acmicpc.net/problem/8393) : n이 주어질 때, 1부터 n까지의 합을 구한다.  
```python
print(sum(range(1, int(input()) + 1)))
```
풀이 : sum(iterable, start = 0) 함수를 알면 쉽다. iterable하고, numeric한 자료형 즉 정수 혹은 실수로 이루어진 리스트나 튜플같은 자료형이 들어오면 내부 요소 합 + start를 더해서 반환한다. start는 default가 0이다.
- 간단한 표현같은 경우, 이렇게 한 줄에 다 써도 되지만 복잡해지면 이해하기 힘들수도 있으므로 그때는 여러줄로 나눠 작성하는 것이 좋을 수도 있다.

## 3-4. 25304번, 영수증
[prob 25304](https://www.acmicpc.net/problem/25304) : 매장에서 계산된 영수증 금액이 맞는지 확인한다. 영수증 금액, 물건의 종류 수, 각 물건의 가격과 개수가 차례대로 주어진다.
```python
x = int(input())    # 영수증 금액
n = int(input())    # 물건 종류
sum_price = 0
for i in range(n):
    a, b = map(int, input().split())
    sum_price += a * b
if x == sum_price:
    print("Yes")
else:
    print("No")
```
풀이 : 문제 주어진대로 따라가면 된다.

## 3-5. 15552번, 빠른 A+B
[prob 15552](https://www.acmicpc.net/problem/15552) : 반복문을 사용할 때, 입출력 방식이 느리면 손해가 커질 수 있다. 반복문 사용시 입출력 함수를 input()보다 빠르게 작성해본다. 테스트 케이스 T번만큼 A + B를 출력한다.
```python
# when using loop, use sys.stdin.readline() for speed
# Be careful of using it, it contains \n. type casting needed
import sys
T = int(sys.stdin.readline())
for i in range(T):
    a, b = map(int, sys.stdin.readline().split())
    print(a + b)
```
풀이 : input()보다 sys.stdin.readline()이 더 빠르다. 단, input()과 다르게 \n까지 입력되므로, 문자열에 저장하려면 .rstrip()을 추가로 해줘야 한다.
- sys.stdin.readline()이 길기 때문에 input = sys.stdin.readline 이렇게 변수에 함수명을 담아놓고 나면 편하게 쓸 수 있다.
> sys 모듈은 인터프리터에 의해 사용되거나 유지되는 일부 변수와 인터프리터와 강하게 상호 작용하는 함수에 대한 액세스를 제공합니다. - 점프 투 파이썬  
> 
> stdin은 모든 대화식 입력에 사용된다. (키보드 입력, 파일 등)
- input 호출 시 인자로 주어진 Prompt 문자열 화면에 출력 후, 키보드에서 키를 하나씩 누르면 대응하는 데이터가 버퍼에 들어간 후 \n이 들어올 시 입력 종료
- stdin.readline()과 input() 간의 속도 차이는 Prompt 출력여부와 한번에 읽어와 버퍼에 저장하냐 ( stdin.readline() ), 누를 때마다 버퍼에 보관하냐 ( input() ) 에서 갈린다. ([출처](https://green-leaves-tree.tistory.com/12))

## 3-6. 11021번, A+B - 7
[prob 11021](https://www.acmicpc.net/problem/11021) : 테스트 케이스 개수가 주어지고, 각 케이스 마다 A, B에 대해 "Case #x: "를 출력한 다음, A+B를 출력한다. (테스트 번호는 1부터 시작)
```python
T = int(input())
for i in range(1, T+1):
    a, b = map(int, input().split())
    print(f"Case #{i}: {a + b}") # f-string : inside {}, can use variable
```
풀이 : f-string 사용시, { }안에 변수를 사용하면 변수값을 자동으로 출력해준다.

## 3-7. 11022번, A+B - 8
[prob 11022](https://www.acmicpc.net/problem/11022) : 테스트 케이스 개수가 주어지고, 각 케이스 마다 A, B에 대해 "Case #x: A + B = C"를 출력한다. (테스트 번호는 1부터 시작)
```python
T = int(input())
for i in range(1, T+1):
    a, b = map(int, input().split())
    print(f"Case #{i}: {a} + {b} = {a + b}")
```
풀이 : 11021번 (바로 위 문제)와 동일하다.

## 3-8. 2438번, 별 찍기 - 1
[prob 2438](https://www.acmicpc.net/problem/2438) : 첫째 줄에는 별 1개, 둘째 줄에는 별 2개, N번째 줄에는 별 N개를 찍는 문제이다.
```python
n = int(input())
for i in range(1, n+1):
    print("*" * i) # when using python, can multiplying a number at string
```
풀이 : 문자열에 정수를 곱하면 정수 크기만큼 반복해서 붙인다.

## 3-9. 2439번, 별 찍기 - 2
[prob 2439](https://www.acmicpc.net/problem/2439) : 첫째 줄에는 별 1개, 둘째 줄에는 별 2개, N번째 줄에는 별 N개를 찍는 문제이다. 하지만, 오른쪽을 기준으로 정렬한 별을 출력해야 한다.
```python
n = int(input())
for i in range(1, n+1):
    print(" " * (n-i), end = '')
    print("*" * i)
```
풀이 : 2438 (바로 위 문제)와 거의 동일하다. print("~~", end = "")처럼 print() 사용 후 한줄 띄우지 않도록 하는 것만 알고 있으면 된다.

## 3-10. 10871번, X보다 작은 수
[prob 10871](https://www.acmicpc.net/problem/10871) : 정수 N개로 이루어진 수열 A와 정수 X가 주어질 때, A에서 X보다 작은 수를 모두 출력한다.
```python
n, x = map(int, input().split())
a = list(map(int, input().split()))
for i in a:
    if i < x:
        print(i, end=' ')
```
풀이 : list 자료 구조를 활용하기 위해 map(int, input().split())에 list()를 씌워준다.  
***
위와 같은 상황에서 반복문을 사용할 시, index를 i에 담는게 아니라, 요소 자체를 담는 게 더 효율적이고 깔끔하다. 
```python
for i in range(n):
    if a[i] < x:
        print(a[i], end = ' ')
```
위의 코드보다
```python
for i in a:
    if i < x:
        print(i, end=' ')
```
아래 코드가 더 낫다. 위의 코드는 파이썬을 c처럼 짜는 것 같고 아래가 파이썬을 잘 활용하는 것 같다.

## 3-11. 10952번, A+B - 5
[prob 10952](https://www.acmicpc.net/problem/10952) : 여러 테스트가 진행되고 각 테스트는 A와 B가 주어질 때, A+B를 각 테스트마다 출력한다. 입력의 마지막에 "0 0"이 입력된다.
```python
while 1:
    a, b = map(int, input().split())
    if a == 0 and b == 0:
        break
    print(a + b)
```
풀이 : 종료 조건에 맞게 break 해주면 된다. while에 해당하는 조건이 언제 올 지 특정하기 어려울 때 1로 두고 while문 안에서 break를 설정하면 된다.

## 3-12. 10951번, A+B - 4
[prob 10951](https://www.acmicpc.net/problem/10951) : 여러 개의 테스트 케이스가 있고, 각 테스트 케이스에 A와 B가 주어질 때, A+B를 출력한다. 입력이 멈출 때까지 테스트를 계속한다.
```python
while 1:
    try:
        a, b = map(int, input().split())
        print(a + b)
    except:
        break
```
풀이 : try except를 사용한 경우, try에서 에러가 발생하면 except로 넘어가서 코드를 실행한다.

## 3-13. 1110번, 더하기 사이클
[prob 1110](https://www.acmicpc.net/problem/1110) : 0보다 크거나 같고, 99보다 작거나 같은 정수가 주어질 때 다음과 같은 연산을 할 수 있다. 먼저 주어진 수가 10보다 작다면 앞에 0을 붙여 두 자리 수로 만들고, 각 자리의 숫자를 더한다. 그 다음, 주어진 수의 가장 오른쪽 자리 수와 앞에서 구한 합의 가장 오른쪽 자리 수를 이어 붙이면 새로운 수를 만들 수 있다. 위의 연산을 처음 주어진 수로 돌아올 때까지 반복할 때, 반복 횟수를 출력한다.
```python
n = int(input())
org_n = n
cnt = 0
while 1:
    L = int(n / 10)
    R = n % 10
    sum = L + R
    sum_R = sum % 10
    new_n = 10 * R + sum_R
    cnt += 1
    if new_n == org_n:
        print(cnt)
        break
    n = new_n
```
풀이 : 복잡해 보이지만 문제에서 제시한 연산을 차례대로 따라가면 된다.

# step3 : 조건문 요약
for는 반복횟수가 선형적이거나, iterable한 자료구조에 차례대로 접근할 때 사용하면 좋다. while은 반복 횟수를 선형적으로 특정하기 어려울 때 사용하면 좋다.