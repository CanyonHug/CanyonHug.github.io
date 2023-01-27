---
title: "Baekjoon problem step9 sorting (정렬)"
excerpt: "Step9 정렬 풀이집입니다."
date: 2022-11-01
categories:
    - Algorithm
tags:
    - Baekjoon
last_modified_at: 2022-11-03
---
- Step9 정렬 풀이집입니다.

# sorting, 정렬
정렬 - [step 9 문제 리스트](https://www.acmicpc.net/step/9)  
배열의 원소를 특정 순서대로 나열하는 방법을 배웁니다.  

## 9-1. 2750번, 수 정렬하기
[prob 2750](https://www.acmicpc.net/problem/2750) : N개의 수를 오름차순으로 정렬한다. O(n^2) 알고리즘으로 가능하다. ex) 삽입(insertion), 거품(bubble) 정렬
```python
n = int(input())
nums = []
for i in range(n):
    nums.append(int(input()))
nums.sort()
for i in nums:
    print(i)
```
풀이 1 : python의 list.sort() 함수를 이용해서 정렬하는 법이다.
- list.sort()는 Timsort 알고리즘을 사용한다. 최선이 O(n), 최악이 O(nlogn)이다. Timsort에 대한 설명은 (https://questionet.tistory.com/61)에서 추가로 확인할 수 있다.

```python
# 수 정렬하기 - using insertion sorting
n = int(input())
nums = []
for i in range(n):
    nums.append(int(input()))

for end in range(1, len(nums)):
    for i in range(end):
        if nums[end] < nums[i]:
            nums[end], nums[i] = nums[i], nums[end]

for i in nums:
    print(i)
```
풀이 2 : 삽입정렬을 이용해 정렬하는 법이다. 삽입정렬은 리스트의 왼쪽부터 오른쪽으로 가며 정렬하는 방식이다. 왼쪽이 정렬된 부분이고, 오른쪽 요소를 정렬된 왼쪽 중 맞는 위치에 끼워넣는다 해서 삽입정렬이다. O(n^2)의 시간복잡도를 갖는다.

![insertion sort](/assets/images/baekjoon_2750_insertion.gif){: width="30%" height="30%" .align-center}
insertion sort from visualgo.net
{: .text-center}

```python
# 수 정렬하기 - using bubble sorting
n = int(input())
nums = []
for i in range(n):
    nums.append(int(input()))
for i in range(len(nums) - 1, 0, -1):
    for j in range(i):
        if nums[j] > nums[j + 1]:
            nums[j], nums[j + 1] = nums[j + 1], nums[j]

for i in nums:
    print(i)
```
풀이 3 : bubble sort를 사용해서 정렬하는 법이다. 거품정렬은 리스트의 오른쪽이 정렬된 부분이다. 리스트의 처음부터 정렬되어 있는 부분까지 이동하면서 맞닿은 두 요소가 크기가 반대면 바꿔준다. 이 때문에 bubble이라고 불린다. 시간복잡도는 O(n^2)이다.

![bubble sort](/assets/images/baekjoon_2750_bubble.gif){: width="30%" height="30%" .align-center}
bubble sort from visualgo.net
{: .text-center}


- bubble sort와 insertion sort의 정렬 방향 혹은 정렬 순서를 반대로 할 수도 있다.
- 정렬 알고리즘을 이미지로 쉽게 확인할 수 있도록 해주는 사이트 : <https://visualgo.net/en/sorting>


## 9-2. 2587번, 대표값2
[prob 2587](https://www.acmicpc.net/problem/2587) : 다섯 개의 자연수가 주어질 때, 평균과 중앙값을 구한다. (중앙값 : 크기 순으로 늘어놓을 때 중앙에 놓인 값)
```python
import sys
input = sys.stdin.readline

num = []
for i in range(5):
    num.append(int(input()))
print(sum(num)//5)
print(sorted(num)[2])
```
풀이 : 리스트의 합으로 평균을 구하고, 정렬된 리스트 중간으로 중앙값을 구한다.

## 9-3. 25305번, 커트라인
[prob 25305](https://www.acmicpc.net/problem/25305) : 응시자 수 N명 중 k명이 상을 받을 때, 커트라인 점수를 출력한다.

```python
import sys
input = sys.stdin.readline

N, k = map(int, input().split())
num = list(map(int, input().split()))
num.sort(reverse = True)
print(num[k - 1])
```
풀이 : 리스트를 내림차순으로 정렬해 k번째 학생의 점수를 출력한다.


## 9-4. 2751번, 수 정렬하기2
[prob 2751](https://www.acmicpc.net/problem/2751) : N개의 수가 주어질 때, 오름차순으로 정렬해 출력한다. O(nlogn)인 정렬 알고리즘으로 풀어야 한다. ex) 병합, 힙 정렬
  
```python
# 수 정렬하기 2
import sys
n = int(input())
nums = []
for i in range(n):
    nums.append(int(sys.stdin.readline()))
nums.sort()
for i in nums:
    print(i)
```
풀이 1 : 2750번에서 풀었던 것처럼 파이썬 sort 내장 함수는 Timsort, nlogn의 시간복잡도를 가지기 때문에 그냥 사용가능하다.

```python
# 수 정렬하기 2 using merge sorting
def merge(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2

    L = merge(arr[:mid])
    R = merge(arr[mid:])

    i, j = 0, 0
    arr_sorted = []
    while i < len(L) and j < len(R):
        if L[i] < R[j]:
            arr_sorted.append(L[i])
            i += 1
        else:
            arr_sorted.append(R[j])
            j += 1
    arr_sorted += L[i:]
    arr_sorted += R[j:]
    return arr_sorted
    
import sys
n = int(input())
nums = []

for i in range(n):
    nums.append(int(sys.stdin.readline()))

nums = merge(nums)

for i in nums:
    print(i)
```
풀이 2 : merge sort (병합 정렬)을 이용한 풀이 방법이다. 

- merge sort는 리스트를 1이 될 때까지 분할 후, 합치는 과정으로 이루어져있다. 분할의 깊이가 logn, 각 단계에서 병합이 n만큼 소요되기 때문에 nlogn만큼의 시간이 걸린다.

![merge sort](/assets/images/baekjoon_2751_merge.gif){: width="30%" height="30%" .align-center}
merge sort from visualgo.net
{: .text-center}

## 9-5. 10989번, 수 정렬하기3
[prob 10989](https://www.acmicpc.net/problem/10989) : N개의 수가 주어질 때, 오름차순으로 출력한다.
```python
# 수 정렬하기 3
import sys
n = int(input())
check = [0] * 10001
for i in range(n):
    check[int(sys.stdin.readline())] += 1
for i in range(10001):
    if check[i] != 0:
        for j in range(check[i]): # repeated number
            print(i)
```
풀이 : 위의 문제들처럼 반복문으로 리스트에 값을 추가한 후에, sort함수를 쓰면 쉽게 풀리지만, 이 문제에는 공간 복잡도 제한이 걸려있다. 따라서 큰 수일 가능성이 있는 수를 배열에 직접 추가하기보다, 배열 값을 인덱스로 활용해 개수를 카운트하는 방식을 사용한다.


## 9-6. 2108번, 통계학
[prob 2108](https://www.acmicpc.net/problem/2108) : 홀수인 N개의 수에 대해 산술평균(arithmetic mean), 중앙값(median), 최빈값(mode), 범위(range)를 출력한다.
```python
# 통계학, arithmetic mean, median, mode, range
import sys
from collections import Counter
n = int(sys.stdin.readline())

nums = []
for i in range(n):
    nums.append(int(sys.stdin.readline()))
nums.sort()
print(round(sum(nums) / n))   # arithmetic mean

print(nums[n // 2])     # median

cnt = Counter(nums).most_common(2)  # mode
if len(cnt) > 1:
    if cnt[0][1] == cnt[1][1]:  # two mode number exists
        print(cnt[1][0])
    else:
        print(cnt[0][0])
else:
    print(cnt[0][0])

print(nums[-1] - nums[0])
```
풀이 : 각 정의에 맞게 출력한다.
- round()함수는 반올림을 해준다.
- collections 모듈의 Counter 객체는 생성자에 넘겨진 인자의 원소가 각각 몇 번씩 나오는지를 dictionary처럼 담고 있다.  
> Counter(["hi", "hey", "hi", "hi", "hello", "hey"]) 아래처럼 Counter 객체 생성  
Counter({'hi': 3, 'hey': 2, 'hello': 1})
- Counter의 most_common(n) 메소드는 최빈값 n개를 리스트에 담긴 튜플 형태로 반환한다.



## 9-7. 1427번, 소트인사이드
[prob 1427](https://www.acmicpc.net/problem/1427) : 주어진 수의 각 자리수를 내림차순으로 정렬해 출력한다.  
```python
n = str(input())
print("".join(sorted(n, reverse = True)))
```
풀이 : int로 받지 않고 string으로 받아서 내림차순으로 정렬한 후, 리스트를 다시 합쳐서 출력한다.
- sorted()의 반환형은 list이다.
- ''.join(list)는 list의 각 원소들을 합쳐서 문자열로 반환해준다.
- '구분자'.join(list)는 list의 각 원소 사이에 구분자를 넣어서 출력해준다.  
ex) '_'.join([a, b, c])는 'a_b_c'이다.


## 9-8. 11650번, 좌표 정렬하기
[prob 11650](https://www.acmicpc.net/problem/11650) : 2차원 평범 위 점 N개가 주어질 때, 좌표를 x좌표 증가하는 순서로, x 좌표 같으면 y좌표 증가 순으로 출력한다.
```python
# 좌표 정렬하기
n = int(input())
coordinate = []
for i in range(n):
    coordinate.append(tuple(map(int, input().split())))
coordinate.sort()
# same with coordinate.sort(key = lambda x: (x[0], x[1]))
for i in coordinate:
    print(i[0], i[1])
```
풀이 : 튜플을 원소로 갖는 리스트에 sort를 취하면 튜플의 1번째 우선으로 정렬이 된다.
- 파이썬의 sort함수 사용시, reverse와 key 두가지 파라미터가 있다.
- reverse = True로 설정하면 내림차순이다.
- key 파라미터는 함수를 저장받고, key 함수 값를 기준으로 정렬한다. 주로 key 함수에 lambda함수로 작성한다.
- key 함수에 lambda 작성 시, 표현식을 튜플로 작성하면 1번째 우선으로 정렬해준다.
- **labmda 매개변수 : 표현식** 으로 작성한다. 추가 설명은 다음에서 확인할 수 있다. <https://dojang.io/mod/page/view.php?id=2359>, <https://wikidocs.net/64>

## 9-9. 11651번, 좌표 정렬하기2
[prob 11651](https://www.acmicpc.net/problem/11651) : 2차원 좌표 N개를 y좌표 증가 순, y 좌표 같을 시 x좌표 증가순으로 출력한다.
```python
# 좌표 정렬하기 2
n = int(input())
coordinate = []
for i in range(n):
    coordinate.append(tuple(map(int, input().split())))
coordinate.sort(key = lambda x: (x[1], x[0]))
for i in coordinate:
    print(i[0], i[1])
```
풀이 : 위의 문제와 같으나, lambda 함수 표현식 순서를 (x[0], x[1])에서 (x[1], x[0])으로 바꿔준다. (우선 순위 변화)

## 9-10. 1181번, 단어 정렬
[prob 1181](https://www.acmicpc.net/problem/1181) : 알파벳 소문자로 이뤄진 N개의 단어를 1. 길이가 짧은 것부터, 2. 길이 같으면 사전 순으로 정렬한다. (같은 단어 여러 번 입력되면 한 번만 출력)
```python
# 단어 정렬
n = int(input())
word_set = set()
for i in range(n):
    word_set.add(input())
word_set = sorted(list(word_set), key = lambda x: (len(x), x))
for i in word_set:
    print(i)
```
풀이 : set()을 이용해 중복을 없애준 후, 정렬할 때 lambda 함수 표현식 튜플로 len(x)와 x를 사용해 정렬 우선순위를 적용해 준다.


## 9-11. 10814번, 나이순 정렬
[prob 10814](https://www.acmicpc.net/problem/10814) : N명의 나이와 이름이 주어질 때, 나이 1순위 가입 순서 2순위로 정렬해서 출력한다.
```python
# 나이순 정렬
n = int(input())
people = []
for i in range(n):
    age, name = input().split()
    age = int(age)
    people.append((age, name))
people.sort(key = lambda x : x[0])
for i in people:
    print(i[0], i[1])
```
풀이 : lambda 함수로 나이 순으로 정렬해준다. 보통 정렬 함수들은 정렬 기준에서 같을 때 원래 순서를 보존하기 때문에 가입 순서는 고려하지 않아도 된다.

## 9-12. 18870번, 좌표 압축
[prob 18870](https://www.acmicpc.net/problem/18870) : 수직선 위의 N개의 좌표에 대해 각 좌표를 그 좌표보다 작은 서로 다른 좌표의 개수로 바꿔서 좌표 압축을 한다. 이때 압축된 결과를 출력한다.
```python
# 좌표 압축
import sys
n = int(sys.stdin.readline())
nums = list(map(int, sys.stdin.readline().split()))
temp = list(sorted(set(nums)))
dic = {temp[i] : i for i in range(len(temp))}

for i in nums:
    print(dic[i], end = ' ')
# same with print(*[dic[i] for i in nums])
# print(*[]) : print list without '[]' and ','.
```
풀이 : 각 좌표보다 작은 서로 작은 좌표의 개수를 구해야 하기 때문에, set()으로 중복 제거 후, 정렬하면 각 좌표의 index가 좌표 압축 결과이다. index를 활용하기 위해 dictionary로 데이터를 만든 후 출력한다.
- list나 dictionary 생성 시, for을 사용해서 한 줄에 생성 가능하다.
- print(*list) 사용 시, list의 원소를 한 칸씩 띄워서 출력해준다.  
ex) [a, b, c]를 a b c 로 출력해준다.


# step9 : 정렬 요약
sorting에서는 입력받은 데이터들을 정렬 기준에 맞게 정렬하여 문제해결에 활용하는 방법을 배웠다. 주로 list.sort(), sorted(iterable) 같은 함수를 사용한다. sort, sorted 사용 시, key 파라미터 lambda 함수로 정렬 기준을 만들어 정렬 가능하다. 경우에 따라 중복 제거 시에는 set()을 활용하는 것도 유용하다. 