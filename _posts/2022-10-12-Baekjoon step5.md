---
title: "Baekjoon problem step5 Function (함수)"
excerpt: "Step5 함수 풀이집입니다."
date: 2022-10-12
categories:
    - Algorithm
tags:
    - Baekjoon
last_modified_at: 2022-10-12
---
- Step5 함수 풀이집입니다.

# Function, 함수
함수을 사용해보기 - [step 5 문제 리스트](https://www.acmicpc.net/step/5)  
특정 동작을 수행하는 부분을 함수로 만들어 관리하는 방법을 배웁니다.  

## 5-1. 15596번, 정수 N개의 합
[prob 15596](https://www.acmicpc.net/problem/15596) : 정수 n개가 주어질 때, n개의 합을 구하는 함수를 작성한다.
 - def solve(a: list) -> int 
 - a : 합을 구해야 하는 정수 n개가 저장되어 있는 리스트
```python
def solve(a: list) -> int:
    return sum(a)
```
풀이 : 함수에 쓰인 ':' 콜론은 매개변수 a가 리스트 데이터 타입이라는 것을 알려주는 주석이다. 또한 '->' 화살표는 함수 리턴 값의 데이터 타입을 알려주는 주석이다. 주석이기 때문에 다른 데이터 타입이 들어오거나 리턴되어도 문제가 없다.
- 출처 : <https://devpouch.tistory.com/189>

## 5-2. 4673번, 셀프 넘버
[prob 4673](https://www.acmicpc.net/problem/4673) : 셀프 넘버는 1949년 인도 수학자 D.R. Kaprekar가 이름 붙였다. 양의 정수 n에 대해서 d(n)을 n과 n의 각 자리수를 더하는 함수라고 정의하자. 예를 들어, d(75) = 75+7+5 = 87이다.

양의 정수 n이 주어졌을 때, 이 수를 시작해서 n, d(n), d(d(n)), d(d(d(n))), ...과 같은 무한 수열을 만들 수 있다.  
n을 d(n)의 생성자라고 한다. 생성자가 없는 숫자를 셀프 넘버라고 한다. 100보다 작은 셀프 넘버는 총 13개가 있다. 1, 3, 5, 7, 9, 20, 31, 42, 53, 64, 75, 86, 97

10000보다 작거나 같은 셀프 넘버를 한 줄에 하나씩 증가하는 순서로 출력하는 프로그램을 작성한다.

```python
num = set(range(1, 10001))
generated_num = set()

for i in range(1, 10001):
    for j in str(i):
        i += int(j)
    generated_num.add(i)

self_num = sorted(num - generated_num)
for i in self_num:
    print(i)
```
풀이 : 처음에는 생성자를 알아내기 위해 역함수를 생각했으나 복잡해져서 접었다. 그래서 생성자들을 걸러내는 방법을 생각했다. 
- 중복을 제거할 때 set을 활용하는 게 좋다. set 빼기를 통해 차집합을 구할 수 있다.
- 처음엔 순서가 존재하지 않는 set에 대해 sorted()를 수행하는게 이해가 되지 않았는데, sorted()의 리턴은 정렬된 list이기 때문에 가능한 것으로 보인다.

## 5-3. 1065번, 한수
[prob 1065](https://www.acmicpc.net/problem/1065) : 어떤 양의 정수 X의 각 자리가 등차수열을 이룬다면, 그 수를 한수라고 한다. 등차수열은 연속된 두 개의 수의 차이가 일정한 수열을 말한다. N이 주어졌을 때, 1보다 크거나 같고, N보다 작거나 같은 한수의 개수를 출력하는 프로그램을 작성하시오. 
  
```python
n = int(input())
cnt = 0
for i in range(1, n+1):
    if i < 100:
        cnt += 1
    else:
        a = str(i)[0]
        diff = int(a) - int(str(i)[1])
        for j in str(i)[1:]:
            if int(a) - int(j) != diff:
                cnt -= 1
                break
            a = j
        cnt += 1
print(cnt)
```
풀이 : 100보다 작은 경우, 항상 자릿수끼리 등차수열이 성립하고, 100 이상인 경우, 자릿수를 늘려가며 등차가 모두 같은지 확인한다.

# step5 : 함수 요약
함수를 제대로 활용한 문제는 딱히 없었던 것 같다. 오히려 이후 step에서 함수를 활용할 수 있을 것 같다.