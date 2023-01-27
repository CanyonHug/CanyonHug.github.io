---
title: "Baekjoon problem step15 back-tracking (백트래킹)"
excerpt: "Step15 백트래킹 풀이집입니다."
date: 2022-11-08
categories:
    - Algorithm
tags:
    - Baekjoon
last_modified_at: 2022-11-08
---
- Step15 백트래킹 풀이집입니다.

# back-tracking, 백트래킹
백트래킹 - [step 15 문제 리스트](https://www.acmicpc.net/step/15)  
모든 경우를 탐색하는 백트래킹 알고리즘을 배웁니다.  
- back-tracking : 해를 찾는 도중 해가 아니어서 막힐 경우, 되돌아가서 다시 해를 찾는 기법 (DFS를 사용하여 풀 수 있는데, 일반적인 DFS와의 차이점은 유망하지 않은 case를 탐색 중지하고 이전 노드로 돌아간다는 점이다.)
- DFS는 주로 스택을 활용해서 구현한다. LIFO의 특성을 활용한다. 재귀함수도 스택처럼 사용되기 때문에 재귀함수로도 구현한다.

## 15-1. 15649번, N과 M (1)
[prob 15649](https://www.acmicpc.net/problem/15649) : 1부터 N까지 자연수 중, 중복 없이 M개를 고른 수열을 모두 출력한다. (수열은 사전 순으로 증가하는 순서로 출력한다.)
```python
# N과 M (1) using back-tracking
n, m = map(int, input().split())

def f(s: list):
    if len(s) == m:
        print(' '.join(map(str, s)))
        return
    
    for i in range(1, n+1):
        if i in s:
            continue
        f(s + [i])
f([])
```
```python
# N과 M (1) using back-tracking
n, m = map(int, input().split())
s = []
def f():
    if len(s) == m:
        print(' '.join(map(str, s)))
        return
    
    for i in range(1, n+1):
        if i in s:
            continue
        s.append(i)
        f()
        s.pop()
f()
```
풀이 1 : 빈 리스트에 1부터 N까지 중 하나를 추가해나가며 DFS를 수행한다. 중복 순열이 아니기 때문에 리스트에 이미 존재할 시, continue 해준다.
- DFS 함수 구현할 때, 인자로 리스트를 전달하지 않는 경우, append와 pop을 함수 앞 뒤로 추가해준다.
- 파이썬에서 함수 인자 작성시, 데이터 타입에 대한 힌트를 작성하기 위해 '인자명: 데이터타입', '함수명(인자) -> 리턴 타입': 으로 표시해줄 수 있다. 하지만, 코드 상에서 아무런 강제 효과가 없기에 주석처럼 받아들이면 된다.

```python
# N과 M (1) using permutations from itertools
from itertools import permutations

n, m = map(int, input().split())
P = permutations(range(1, n+1), m)
for i in P:
    print(' '.join(map(str, i)))
```
풀이 2 : itertools 모듈의 permutations 클래스를 쓰면 쉽게 순열을 구할 수 있다.
- permutations(iter, int num) : int 형 iterable 데이터에서 num만큼 고른 순열을 저장한다.

## 15-2. 15650번, N과 M (2)
[prob 15650](https://www.acmicpc.net/problem/15650) : 자연수 N과 M이 주어질 때, 1부터 N까지의 자연수 중 중복 없이 M개를 고른 오른차순 수열을 구한다.
```python
# N과 M (2) using DFS
n, m = map(int, input().split())

def f(s: list, start: int):
    if len(s) == m:
        print(*s)
        return
    
    for i in range(start, n + 1):
        if i not in s:
            f(s + [i], i)
    
f([], 1)
```
풀이 1 : 위의 문제와 유사하나 수열이 오름차순이어야 한다. 앞 뒤 순서가 오름차순으로 정해져 있기 때문에 조합의 경우의 수와 같다고 볼수도 있다.

```python
# N과 M (2) using combinations from itertools
from itertools import combinations

n, m = map(int, input().split())
C = combinations(range(1, n+1), m)
for i in C:
    print(' '.join(map(str, i)))
```
풀이 2 : 순서가 오름차순이기 때문에 combination을 사용해서 풀 수 있다.

## 15-3. 15651번, N과 M (3)
[prob 15651](https://www.acmicpc.net/problem/15651) : 1부터 N까지의 자연수 중 M개를 중복을 허용해서 고른 수열을 출력한다. (수열은 증가하는 사전 순으로 출력한다.)
```python
# N과 M (3) using DFS
n, m = map(int, input().split())

def f(s: list):
    if len(s) == m:
        print(*s)
        return

    for i in range(1, n + 1):
        f(s + [i])
f([])
```
풀이 1 : 중복순열이기 때문에 리스트 안에 존재하는지 확인하지 않아도 된다.

```python
# N과 M (2) using product from itertools
from itertools import product

n, m = map(int, input().split())
P = product(range(1, n+1), m)
for i in P:
    print(' '.join(map(str, i)))
```
풀이 2 : 중복순열은 itertools 모듈의 product 클래스를 활용해서 구할 수 있다. (백준에서는 런타임에러로 통과하지 못한다. DFS 방식이 더 빠른 것 같다.)

## 15-4. 15652번, N과 M (4)
[prob 15652](https://www.acmicpc.net/problem/15652) : 1부터 N까지의 자연수 중 M개를 중복을 허용한 비내림차순인 수열을 출력한다. (중복조합과 같다.)
```python
# N과 M (4) using DFS
n, m = map(int, input().split())

def f(s: list, start: int):
    if len(s) == m:
        print(*s)
        return

    for i in range(start, n + 1):
        f(s + [i], i)

f([], 1)
```
풀이 1 : 조합을 구할 때와 마찬가지로 start변수로 내림차순을 방지하지만, 중복을 허용하도록 구성한다.

```python
# N과 M (2) using combinations_with_replacement from itertools
from itertools import combinations_with_replacement

n, m = map(int, input().split())
C = combinations_with_replacement(range(1, n+1), m)
for i in C:
    print(' '.join(map(str, i)))
```
풀이 2 : 중복조합은 itertools 모듈의 combinations_with_replacement 클래스로 구현할 수 있다. (중복순열과는 달리 중복조합은 itertools 쓰고 통과가 된다.)

## 15-5. 9663번, N-Queen
[prob 9663](https://www.acmicpc.net/problem/9663) : N-Queen 문제는 크기가 N × N인 체스판 위에 퀸 N개를 서로 공격할 수 없게 놓는 문제이다. N이 주어졌을 때, 퀸을 놓는 방법의 수를 구하는 프로그램을 작성하시오.
```python
# N-Queen
# Python3 can not be accepted in Baekjoon. use pypy3
n = int(input())
cnt = 0
row = [0] * n

def pos_possible(a):
    for i in range(a):
        if row[a] == row[i] or abs(row[a] - row[i]) == abs(a - i):
            return False
    return True

def N_Queens(a):
    global cnt
    if a == n:
        cnt += 1
        return
    
    for i in range(n):
        row[a] = i
        if pos_possible(a): # If row[a] = i is possible, move to row[a+1]
            N_Queens(a + 1)
N_Queens(0)
print(cnt)
```
풀이 : 퀸을 둘 수 있는 경우는 다른 퀸과 일직선상, 대각선상에 없는 경우 가능하다. 이를 check하는 함수를 작성해주고, DFS를 실행하는 함수를 작성해준다. row 리스트는 index가 행, 값이 열을 의미해서 (index, value)로 체스판 위 좌표를 나타낸다.
- 함수 안에서 변수 앞에 global을 붙이면 전역 변수로 사용가능하다. cnt값은 코드 전체에서 활용되기 때문에 전역변수로 사용해준다.
- 전역 변수는 프로그램을 혼란스럽게 만들기 때문에 가급적 사용을 지양하라고 배웠으나 불가피한 경우는 사용하면 유용하게 쓸 수 있다.
- python3로는 런타임을 통과하지 못하는 듯하다. pypy3로 돌릴 시 통과가 된다.


## 15-6. 2580번, 스도쿠
[prob 2580](https://www.acmicpc.net/problem/2580) : 스도쿠 판을 채워서 출력한다.
```python
# 스도쿠
import sys
input = sys.stdin.readline
board = []
zeros = []
for i in range(9):
    board.append(list(map(int, input().split())))
    for j in range(9):
        if board[i][j] == 0:
            zeros.append((i, j))

def check_row(r, a):        # check if a is in row x or not
    for i in range(9):
        if board[r][i] == a:
            return False
    return True

def check_col(c, a):        # check if a is in column y or not
    for i in range(9):
        if board[i][c] == a:
            return False
    return True

def check_square(r, c, a):  # check if a is in 3*3 square
    for i in range(3):
        for j in range(3):
            if board[r//3 * 3 + i][c//3 * 3 + j] == a:
                return False
    return True

def DFS(zero_idx):
    if zero_idx == len(zeros):
        for i in range(9):
            print(*board[i])
        exit(0)                     # print only one possible case
    
    for i in range(1, 10):
        r = zeros[zero_idx][0]
        c = zeros[zero_idx][1]

        if check_row(r, i) and check_col(c, i) and check_square(r, c, i):
            board[r][c] = i
            DFS(zero_idx + 1)
            board[r][c] = 0

DFS(0)
```
풀이 : 스도쿠 판을 입력 받으면서 0인 빈칸의 위치 정보도 따로 저장해둔다. 빈칸에 대해 빈칸이 위치한 행, 열, 3 * 3 정사각형을 확인해 1부터 9까지 어떤 수가 들어갈 수 있는지 확인하고 들어갈 수 있는 수를 집어넣는다. 한 가지 경우만 출력하므로 출력 후 exit(0)로 프로그램을 종료시킨다.
- 위에서 리스트로 DFS를 할 때처럼 append를 board리스트에 수 지정, pop을 board 리스트 0으로 초기화 해주는 것이다.

## 15-7. 14888번, 연산자 끼워넣기
[prob 14888](https://www.acmicpc.net/problem/14888) : N개의 수와 N-1개의 연산자가 주어질 때, 만들 수 있는 식의 결과가 최대, 최소인 것을 구하는 프로그램을 작성한다. (식의 계산은 연산자 우선 순위 무시하고 앞부터 진행한다. 나눗셈은 정수 나눗셈으로 몫만 취한다. 음수를 양수로 나눌 때는 양수로 바꾼뒤 몫을 구하고 음수로 바꾼다.)
```python
# 연산자 끼워넣기
import sys
input = sys.stdin.readline

n = int(input())
nums = list(map(int, input().split()))
oper = list(map(int, input().split()))  # [+, -, *, /]

max_n = -1e9
min_n = 1e9


def DFS(depth, total, plus, minus, multiply, divide):
    global max_n, min_n
    if depth == n:
        max_n = max(total, max_n)
        min_n = min(total, min_n)
        return

    if plus:
        DFS(depth + 1, total + nums[depth], plus - 1, minus, multiply, divide)
    if minus:
        DFS(depth + 1, total - nums[depth], plus, minus - 1, multiply, divide)
    if multiply:
        DFS(depth + 1, total * nums[depth], plus, minus, multiply - 1, divide)
    if divide:
        DFS(depth + 1, int(total / nums[depth]), plus, minus, multiply, divide - 1)


DFS(1, nums[0], oper[0], oper[1], oper[2], oper[3])
print(max_n)
print(min_n)
```
풀이 : 계산하는 함수를 따로 만들어 풀이가 더 복잡했었는데 [이 분의 블로그](https://velog.io/@kimdukbae/BOJ-14888-%EC%97%B0%EC%82%B0%EC%9E%90-%EB%81%BC%EC%9B%8C%EB%84%A3%EA%B8%B0-Python)를 참고했다. 함수 호출할 때 계산을 해주는 방식이 더 깔끔한 것 같다.
- 문제 상황에서 요구하는 음수 나누기 정수를 //로 처리해버리면 -0.2를 -1로 출력해서 문제가 생긴다. int(-a / b)를 해야 문제상황에 맞게 할 수 있다.

## 15-8. 14889번, 스타트와 링크
[prob 14889](https://www.acmicpc.net/problem/14889) : 자연수 짝수 N과 Sij가 주어진다. Sij는 i번과 j번이 같은 팀에 속할 때 팀에 더해지는 능력치이다. i와 j가 같은 팀이면 Sij와 Sji가 팀 능력치에 더해진다. 이때 스타트 팀과 링크 팀의 능력치 차이의 최솟값을 출력한다.
```python
# 스타트와 링크
import sys
input = sys.stdin.readline

n = int(input())
s = []
for i in range(n):
    s.append(list(map(int, input().split())))

team = set(i for i in range(1, n + 1))
team_s = [0 for _ in range(n//2)]
min_diff = 1e9

def cal(team_member):
    score = 0
    for i in range(len(team_member) - 1):
        for j in range(i + 1, len(team_member)):
            score += (s[team_member[i] - 1][team_member[j] - 1] + s[team_member[j] - 1][team_member[i] - 1])
    return score

def DFS(depth):
    global min_diff
    if depth == n // 2:
        min_diff = min(min_diff, abs(cal(team_s) - cal(list(team - set(team_s)))))
        return
    
    for i in range(team_s[depth-1] + 1, n // 2 + 2 +depth):
        if i not in team_s:
            team_s[depth] = i
            DFS(depth + 1)
            team_s[depth] = 0
            
DFS(0)
print(min_diff)
```
풀이 : 팀별 능력치를 계산하는 함수와 팀을 뽑아가는 DFS함수를 구성해서 구해준다. DFS 함수에서 반복문을 효율적으로 돌리기 위해서 depth별로 고를 수 있는 범위를 타이트하게 잡는다.

# step15 : 백트래킹 요약
back-tracking에서는 DFS를 활용해서 스택 구조를 통해 해를 찾지 못한 경우 뒤로 돌아가서 실행한다. 주로 재귀함수를 활용해서 back-tracking을 구현했다. 수열, 조합 같은 문제들은 itertools 모듈의 조합, 순열 클래스들을 활용한 방식으로도 풀 수 있다.