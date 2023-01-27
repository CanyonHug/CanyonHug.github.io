---
title: "Baekjoon problem step12 set and map (집합과 맵)"
excerpt: "Step12 집합과 맵 풀이집입니다."
date: 2022-11-08
categories:
    - Algorithm
tags:
    - Baekjoon
last_modified_at: 2022-11-08
---
- Step12 집합과 맵 풀이집입니다.

# set and map, 집합과 맵
집합과 맵 - [step 12 문제 리스트](https://www.acmicpc.net/step/12)  
집합과 맵을 활용하는 방법을 배웁니다. 특정 원소가 속해 있는지 빠르게 찾거나, 각 원소에 대응되는 원소를 빠르게 찾는 자료구조를 배워 봅시다.	 

## 12-1. 10815번, 숫자 카드
[prob 10815](https://www.acmicpc.net/problem/10815) : 숫자 카드 N개를 가지고 있다. 이때 M개의 정수가 주어질 때, 숫자 카드를 상근이가 가지고 있는지 아닌지를 구하는 프로그램을 작성한다.
```python
# 숫자 카드 using binary search
import sys
input = sys.stdin.readline

n = int(input())
n_nums = list(map(int, input().split()))
m = int(input())
m_nums = list(map(int, input().split()))

n_nums.sort()

def bi_search(element, list, start = 0, end = None):
    if end == None:
        end = len(list) - 1
    if start > end:
        return 0

    mid = (start + end) // 2
    if element == list[mid]:
        return 1
    elif element < list[mid]:
        end = mid - 1
    else:
        start = mid + 1
    
    return bi_search(element, list, start, end)

cnt = []

for i in m_nums:
    cnt.append(bi_search(i, n_nums))

print(*cnt)    
```
풀이 1 : 가지고 있는 n개의 숫자들을 정렬한 후, n개 리스트에서 m개의 숫자들을 각각 binary search를 통해 존재하는지 확인한다.
- binary search : 정렬된 리스트에서 원소가 존재하는지 찾는 알고리즘이다. 리스트의 절반 index를 기준으로 찾는 원소와 대소 비교하며 찾아가기 때문에 binary search이다. 시간복잡도는 O(logn)이다.  

![binary search](/assets/images/baekjoon_10815_binary.gif){: width="50%" height="50%" .align-center}
binary search from visualgo.net
{: .text-center}

```python
# 숫자 카드 using set
import sys
input = sys.stdin.readline

n = int(input())
n_nums = set(map(int, input().split()))
m = int(input())
m_nums = list(map(int, input().split()))

# when checking list : O(n), set : O(1)
# list should be checked all indices
for i in m_nums:        
    if i in n_nums:
        print(1, end = ' ')
    else:
        print(0, end = ' ')
```
풀이 2 : x in arr를 실행할 때, 리스트는 O(n)이 소요되지만, set은 O(1)이 걸리기 때문에 훨씬 효율적이다.
- list : 데이터 수정, 순서 필요한 경우
- tuple : 데이터의 읽기만 필요한 경우
- set : 중복된 값 불허, 순서 불필요한 경우

## 12-2. 14425번, 문자열 집합
[prob 14425](https://www.acmicpc.net/problem/14425) : N개의 문자열 집합 S가 주어질 때, M개의 문자열 중 S에 포함된 것이 몇 개인지 구한다.
```python
# 문자열 집합
import sys
input = sys.stdin.readline

n, m = map(int, input().split())
s_set = set()
cnt = 0
for i in range(n):
    s_set.add(input().rstrip())
for i in range(m):
    if input().rstrip() in s_set:
        cnt += 1
print(cnt)
```
풀이 : 위의 문제와 마찬가지로 set에 존재하는지 확인해서 count 결과를 출력한다.
- input().rstrip()은 문자열의 오른쪽 공백을 잘라준다.
- strip()은 좌우 공백, lstrip()은 왼쪽 공백을 잘라준다.

## 12-3. 1620번, 나는야 포켓몬 마스터 이다솜
[prob 1620](https://www.acmicpc.net/problem/1620) : 도감에 수록된 포켓몬 개수 N과 맞춰야 하는 문제의 개수 M이 주어진다. N개의 포켓몬들의 이름이 주어지고, M개의 포켓몬 번호 혹은 이름이 주어질 때, M개의 문제의 답안을 출력한다. 번호가 주어진 경우, 이름을 출력하고 이름이 주어진 경우 번호를 출력한다.
```python
# 나는야 포켓몬 마스터 이다솜
import sys
input = sys.stdin.readline

n, m = map(int, input().split())
dic = {}
for i in range(1, n + 1):
    dic[i] = input().rstrip()  # num : name
reverse_dic = dict(map(reversed, dic.items()))  # name : num
for i in range(m):
    ask = input().rstrip()
    if ask.isdigit():
        print(dic[int(ask)])
    else:
        print(reverse_dic[ask])
```
풀이 : 문제에서 번호를 물어보는 경우 이름을, 이름을 물어보는 경우 번호를 출력해야 하기 때문에 순서가 필요하다. 따라서 set은 사용 불가하고 list나 dictionary를 사용해야한다. list는 이름이 주어질 경우 찾는 지 확인하는데 O(n)이 걸리기 때문에 dictionary와 reverse dictionary를 활용한다.
- key : value로 이루어진 dictionary에서 value값으로 key를 찾고자 할 때는 dict를 뒤집는 방법과 for 문을 활용하는 방법이 있다.
- dict을 뒤집는 경우, map을 활용해 reversed 함수로 뒤집어 준다.

## 12-4. 10816번, 숫자 카드 2
[prob 10816](https://www.acmicpc.net/problem/10816) : 숫자 카드 N개를 갖고 있고 M개의 정수가 주어질 때, 각각 M개의 숫자를 몇 개 가지고 있는지 구하는 프로그램을 작성한다.
```python
# 숫자 카드 2
import sys
input = sys.stdin.readline

n = int(input())
n_nums = list(map(int, input().split()))
dic = {}
for i in n_nums:
    if i in dic:
        dic[i] += 1
    else:
        dic[i] = 1
m = int(input())
m_nums = list(map(int, input().split()))
for i in m_nums:
    print(dic.get(i, 0), end = ' ')
```
풀이 : N개의 카드들을 key값을 숫자, value 값을 개수로 갖는 딕셔너리로 만들고 M개의 숫자들에 대해 dic.get()을 사용해서 value값을 출력한다.
- dict.get(x, 디폴트값)은 key 값을 x로 가지는 value를 반환하고, 없는 경우 디폴트값을 반환한다.

## 12-5. 1764번, 듣보잡
[prob 1764](https://www.acmicpc.net/problem/1764) : 듣도 못한 N명과, 보도 못한 M명이 주어질 때, 듣도잡의 수와 명단을 사전순으로 출력한다.
```python
# 듣보잡
import sys
input = sys.stdin.readline

n, m = map(int, input().split())
n_name = set([input().rstrip() for i in range(n)])
m_name = set([input().rstrip() for i in range(m)])
result = sorted(list(n_name & m_name))

print(len(result))
for i in result:
    print(i)
```
풀이 : 교집합을 사용하기 위해, n과 m을 set으로 만들고 &으로 교집합연산을 수행한다. 이를 사전순으로 정렬하기 위해 list함수 후, sorted를 사용한다.

## 12-6. 1269번, 대칭 차집합
[prob 1269](https://www.acmicpc.net/problem/1269) : 자연수를 원소로 갖는 set A, B가 있을 때, A, B의 대칭 차집합의 원소 개수를 출력한다. (대칭 차집합은 A-B, B-A의 합집합이다.)
```python
# 대칭 차집합
import sys
input = sys.stdin.readline

a, b = map(int, input().split())
a_set = set(map(int, input().split()))
b_set = set(map(int, input().split()))

print(len(a_set - b_set) + len(b_set - a_set))
```
풀이 : 차집합을 이용해 문제 구성을 그대로 따라가면 된다.

## 12-7. 11478번, 서로 다른 부분 문자열의 개수
[prob 11478](https://www.acmicpc.net/problem/11478) : 문자열 S에 대해 S의 서로 다른 부분 문자열 개수를 출력한다.
```python
# 서로 다른 부분 문자열의 개수
import sys
input = sys.stdin.readline

s = input().rstrip()
substring = []
for i in range(1, len(s) + 1):
    for j in range(0, len(s) + 1 - i):
        substring.append(s[j:j + i])
print(len(set(substring)))
```
풀이 : 이중반복문으로 부분 문자열을 구한다.

# step12 : 집합과 맵 요약
set and map에서는 set과 dictionary 자료 구조를 다루는 방법을 배웠다. set은 데이터 자료의 순서, 중복이 필요 없을 때 사용하고, dictionary는 key값으로 value를 받아오는 작업을 빠르게 할 수 잇다. set과 dictionary 모두 원소를 찾을 때 O(1)로 찾을 수 있다는 장점이 있다.