---
title: "Baekjoon problem step6 String (문자열)"
excerpt: "Step6 문자열 풀이집입니다."
date: 2022-10-13
categories:
    - Algorithm
tags:
    - Baekjoon
last_modified_at: 2022-10-13
---
- Step6 문자열 풀이집입니다.

# String, 문자열
문자열을 사용해보기 - [step 6 문제 리스트](https://www.acmicpc.net/step/7)  
문자열 데이터를 저장하고 다루는 방법을 배웁니다.  

## 6-1. 11654번, 아스키 코드
[prob 11654](https://www.acmicpc.net/problem/11654) : 알파벳 소문자, 대문자, 숫자 0-9중 하나가 주어졌을 때, 주어진 글자의 아스키 코드값을 출력하는 프로그램을 작성한다.
```python
a = input()
print(ord(a))
```
풀이 : ord(문자) - 하나의 문자를 인자로 받고, 해당 문자에 해당하는 아스키 정수를 반환  
chr(정수) - 하나의 정수를 인자로 받고 해당 정수에 해당하는 아스키 문자를 반환

## 6-2. 11720번, 숫자의 합
[prob 11720](https://www.acmicpc.net/problem/11720) : N개의 숫자가 공백 없이 쓰여있을 때, 이 숫자를 모두 합해서 출력한다.

```python
a = input()
n = input()
print(sum(map(int, n)))
```
풀이 : string은 iterable하기 때문에 map으로 하나씩 int로 변환시켜서 sum()을 사용할 수 있다.

## 6-3. 10809번, 알파벳 찾기
[prob 10809](https://www.acmicpc.net/problem/10809) : 알파벳 소문자로만 이루어진 단어 S가 주어질 때, 각각의 알파벳에 대해서, 단어에 포함되어 있는 경우에는 처음 등장하는 위치를, 포함되어 있지 않은 경우에는 -1을 출력한다.  
```python
s = input()
alphabet = list(range(ord('a'), ord('z') + 1))
for i in alphabet:
    print(s.find(chr(i)))
# ord() : character -> ASKII code, chr() : ASKII code -> character
```
풀이 : 아스키 코드 값을 활용해 range()로 리스트를 구성하고 다시 chr()를 활용해 리스트에 알파벳이 존재하는지 찾는다. string.find(찾을 문자)를 사용하면, 찾는 문자가 맨 처음 나타나는 index를 반환하고, 존재하지 않으면 -1을 반환하기 때문에 문제 조건 그대로 사용 가능하다.

## 6-4. 2675번, 문자열 반복
[prob 2675](https://www.acmicpc.net/problem/2675) : 문자열 S를 입력받은 후에, 각 문자를 R번 반복해 새 문자열 P를 만든 후 출력하는 프로그램을 작성하시오. 즉, 첫 번째 문자를 R번 반복하고, 두 번째 문자를 R번 반복하는 식으로 P를 만들면 된다. S에는 QR Code "alphanumeric" 문자만 들어있다.

QR Code "alphanumeric" 문자는 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ\$%*+-./: 이다.
  
```python
n = int(input())    # test case number
for i in range(n):
    repeat, s = input().split()
    repeat = int(repeat)
    a = ""
    for j in s:
        a += j * repeat
    print(a)
```
풀이 : 이전에 문자에 정수를 곱하면 정수 크기만큼 곱해서 붙여주는 것을 배웠던 것을 활용하면 된다. 문자열이 iterable함을 이용하여 for 반복문에서 활용할 수 있음도 알아둬야 한다.

## 6-5. 1157번, 단어 공부
[prob 1157](https://www.acmicpc.net/problem/1157) : 알파벳 대소문자로 된 단어가 주어지면, 이 단어에서 가장 많이 사용된 알파벳이 무엇인지 알아내는 프로그램을 작성하시오. 단, 대문자와 소문자를 구분하지 않는다. 단, 가장 많이 사용된 알파벳이 여러 개 존재하는 경우에는 ?를 출력한다.  
```python
s = input().upper()
s_set = list(set(s))
cnt = []

for i in s_set:
    n = s.count(i)
    cnt.append(n)

if cnt.count(max(cnt)) > 1:
    print('?')
else:
    print(s_set[cnt.index(max(cnt))])
```
풀이 : string.upper()를 통해 전부 대문자로 변환해준다. (에제 출력이 항상 대문자이기 때문에 upper()를 사용했다.) 최대 값을 count해서 여러 개인 경우 구분한다. 

## 6-6. 1152번, 단어의 개수
[prob 1152](https://www.acmicpc.net/problem/1152) : 영어 대소문자와 공백으로 이루어진 문자열이 주어진다. 이 문자열에는 몇 개의 단어가 있을까? 이를 구하는 프로그램을 작성하시오. 단, 한 단어가 여러 번 등장하면 등장한 횟수만큼 모두 세어야 한다.  
```python
a = input().split()
print(len(a))
```
풀이 : split()의 return 타입은 list이다. 단어는 공백으로 띄워서 제공되기 때문에 split으로 개수를 세어주면 된다.

## 6-7. 2908번, 상수
[prob 2908](https://www.acmicpc.net/problem/2908) : 세 자리수 두 개가 있을 때, 한 친구는 숫자를 거꾸로 읽는다. 이때 그 친구가 더 큰 수라고 말할 숫자를 출력한다.
  
```python
a, b = input().split()
a, b = int(a[::-1]), int(b[::-1])
if a > b:
    print(a)
else:
    print(b)
```
풀이1 : a[::-1]와 같이 슬라이스 방식을 뒤에서부터 거꾸로 1개씩 슬라이스하는 방식을 쓰면 간단히 reverse 가능하다.
```python
a, b = map(list, input().split())
a.reverse()
b.reverse()
a, b = ''.join(a), ''.join(b)
a, b = int(a), int(b)
if a > b:
    print(a)
else:
    print(b)
```
풀이2 : 거꾸로 슬라이스하는 방식을 모를 때 사용한 방식이다. 리스트로 만들어 분리한 후, list.reverse()해준 후, join으로 다시 합친다.
- ''.join(iterable[string]) : iterable 데이터 형에 저장된 string들을 합쳐서 string을 반환해 준다.
- '구분자'.join(iterable[string]) : string들을 합칠 때 사이에 구분자를 넣어준다.
  
## 6-8. 5622번, 다이얼
[prob 5622](https://www.acmicpc.net/problem/5622) : 아래 그림과 같은 다이얼 전화기를 사용한다.  
![dial phone](/assets/images/baekjoon_5622.jpg){: width="30%" height="30%" .align-center}  

숫자를 하나를 누른 다음에 금속 핀이 있는 곳 까지 시계방향으로 돌려야 한다. 숫자를 하나 누르면 다이얼이 처음 위치로 돌아가고, 다음 숫자를 누르려면 다이얼을 처음 위치에서 다시 돌려야 한다.

숫자 1을 걸려면 총 2초가 필요하다. 1보다 큰 수를 거는데 걸리는 시간은 이보다 더 걸리며, 한 칸 옆에 있는 숫자를 걸기 위해선 1초씩 더 걸린다.

상근이의 할머니는 전화 번호를 각 숫자에 해당하는 문자로 외운다. 즉, 어떤 단어를 걸 때, 각 알파벳에 해당하는 숫자를 걸면 된다. 예를 들어, UNUCIC는 868242와 같다.

할머니가 외운 단어가 주어졌을 때, 이 전화를 걸기 위해서 필요한 최소 시간을 구하는 프로그램을 작성하시오.

```python
alphabet = ['ABC', 'DEF', 'GHI', 'JKL', 'MNO', 'PQRS', 'TUV', 'WXYZ']
s = input()
sum = 0

for unit in alphabet:
    for i in unit:
        for x in s:
            if x == i:
                sum += alphabet.index(unit) + 3
print(sum)
```
풀이 : 같은 위치에 해당하는 알파벳끼리 묶은 후, 반복문을 통해 알파벳이 존재하는 지 확인후 index로 시간을 구한다.

## 6-9. 2941번, 크로아티아 알파벳
[prob 2941](https://www.acmicpc.net/problem/2941) : 예전에는 운영체제에서 크로아티아 알파벳을 입력할 수가 없었다. 따라서, 다음과 같이 크로아티아 알파벳을 변경해서 입력했다.

![Croatia alphabet](/assets/images/baekjoon_2941.jpg){: width="20%" height="20%" .align-center}

단어가 주어졌을 때, 몇 개의 크로아티아 알파벳으로 이루어져 있는지 출력한다.

dž는 무조건 하나의 알파벳으로 쓰이고, d와 ž가 분리된 것으로 보지 않는다. lj와 nj도 마찬가지이다. 위 목록에 없는 알파벳은 한 글자씩 센다.

```python
s = ['c=', 'c-', 'dz=', 'd-', 'lj', 'nj', 's=', 'z=']
a = input()
for i in s:
    a = a.replace(i, 'a')
print(len(a))
```
풀이 : 크로아티아 알파벳 리스트를 만들고, 반복문을 통해 iterator를 돌며 존재 여부를 확인하고 str.replace(i, 'a')로 크로아티아 알파벳을 모두 한 글자인 a로 바꾸고 길이를 확인한다.
- 반복문에서 iterator를 돌며 확인하는 방법을 떠올리지 못했을 때, 조건문으로 case 나눈 후 일일이 count 했는데, 매우 복잡해졌다.
- string 리스트를 만든 후, 반복문으로 iterator를 돌면서 확인하는 방식을 잘 기억해두면 좋을 것 같다.

## 6-10. 1316번, 그룹 단어 체커
[prob 1316](https://www.acmicpc.net/problem/1316) : 그룹 단어란 단어에 존재하는 모든 문자에 대해서, 각 문자가 연속해서 나타나는 경우만을 말한다. 예를 들면, ccazzzzbb는 c, a, z, b가 모두 연속해서 나타나고, kin도 k, i, n이 연속해서 나타나기 때문에 그룹 단어이지만, aabbbccb는 b가 떨어져서 나타나기 때문에 그룹 단어가 아니다.

단어 N개를 입력으로 받아 그룹 단어의 개수를 출력하는 프로그램을 작성하시오.
  
```python
n = int(input())
cnt = 0
for i in range(n):
    cnt += 1
    s = input()
    check = []
    for j in range(len(s)):
        if s[j] in check and s[j] != s[j-1]:
            cnt -= 1
            break
        elif s[j] not in check:
            check.append(s[j])
print(cnt)
```
풀이 : 단어 내에서 연속한지 확인해야 하기 때문에, check 리스트를 만들어서 check 리스트에 해당 index 문자가 존재하는지 확인하고, 있으면 바로 이전 index와 일치하는지 확인한다. 일치하지 않는다면 그룹 단어가 아니므로 cnt -1을 시행한다. 해당 index 문자가 존재하지 않으면 check 리스트에 추가해준다.

# step6 : 문자열 요약
string 데이터 타입 또한 iterable하기 때문에 for 반복문 활용할 때 iterator 순환하는 방법을 기억해 두면 좋을 것 같다. ord(문자), chr(정수), ''.join()같은 함수들도 알고 있으면 좋을 것 같다.