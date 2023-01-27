---
title: "Baekjoon problem step1 in, out (입출력과 사칙연산)"
excerpt: "Step1 입출력, 사칙연산 풀이집입니다."
date: 2022-10-09
categories:
    - Algorithm
tags:
    - Baekjoon
last_modified_at: 2022-10-09
---
- Step1 입출력, 사칙연산 풀이집입니다.

# In, Out 입출력과 사칙연산
입력, 출력과 사칙연산 (+ - * /)을 연습해보기. - [step 1 문제 리스트](https://www.acmicpc.net/step/1)  
매우 기초적인 단계이고, 파이썬은 입력 받을 때 데이터 타입에 크게 신경쓰지 않아도 때문에 별도의 설명이 필요하지 않을 듯 하다.

## 1-1. 2557번, Hello World
[prob 2557](https://www.acmicpc.net/problem/2557) : Hello World!를 출력하시오.
```python
prin("Hello World!")
```
풀이 : 문자열 출력은 print() 안에 글자들을 " " 혹은 ' ' 로 감싸준 다음 입력하면 된다.
## 1-2. 10718번, We love kriii
[prob 10718](https://www.acmicpc.net/problem/10718) : 두 줄에 걸쳐 "강한친구 대한육군"을 한 줄에 한 번씩 출력한다.
```python
print("강한친구 대한육군\n강한친구 대한육군")
```
풀이 1 : print()로 출력시 한 줄 띄우려면 \n을 써주면 다음줄로 넘어간다.

```python
print("강한친구 ", end = "")
print("대한육군")
print("강한친구 ", end = "")
print("대한육군")
```
- print()는 출력 후 자동으로 줄을 바꿔주기 때문에 만약 print()를 여러 번 쓰면서 줄을 바꾸지 않으려면 print("~~", end = "")로 작성해준다.

```python
print("""\
강한친구 대한육군
강한친구 대한육군\
""")
```
풀이 2 : 여러 줄의 문자열을 출력하려면 """ """ 혹은 ''' ''' 사이에 여러줄을 문자열을 작성해주면 그대로 출력된다. 
- 코드가 너무 길어서 한줄에 작성하지 못할 경우, \ 를 써주면 한 줄에 작성한 것과 같은 효과를 준다.

## 1-3. 1000번, A+B
[prob 1000](https://www.acmicpc.net/problem/1000) : 두 수를 입력받고 합을 출력하는 문제  
```python
a, b = input().split()
a = int(a)
b = int(b)
print(a + b)
```
풀이 : 한 줄에 입력이 여러 개 들어오는 경우, input().spli()을 이용해 a와 b에 각각 대입할 수 있다.  
- 또한, a와 b가 입력받는 경우 문자로 인식되기 때문에 더해주기 위해 int()로 정수형으로 바꿔준 뒤 더해준다.

## 1-4. 1001번, A-B
[prob 1001](https://www.acmicpc.net/problem/1001) : 두 수를 입력받고 뺄셈을 한 결과를 출력하는 문제  
```python
a, b = input().split()
a = int(a)
b = int(b)
print(a - b)
```
풀이 생략

## 1-5. 10098번, A*B
[prob 10098](https://www.acmicpc.net/problem/10098) : 곱셈 문제  
```python
a, b = input().split()
a = int(a)
b = int(b)
print(a * b)
```
풀이 생략


## 1-6. 1008번, A/B
[prob 1008](https://www.acmicpc.net/problem/1008) : 나눗셈 문제
```python
a, b = input().split()
print(int(a)/int(b))
```
풀이 생략

## 1-7. 10869번, 사칙연산
[prob 10869](https://www.acmicpc.net/problem/10869) : 모든 연산 문제  
```python
a, b = input().split()
a = int(a)
b = int(b)
print(a + b)
print(a - b)
print(a * b)
print(int(a / b))   # 혹은 // 사용
print(a % b)
```
풀이 : 몫을 출력하기 위해 int(a/b)로 나머지를 없애주거나 //를 사용해준다. 나머지를 구할 때는 % 기호를 사용해주면 된다.

## 1-8. 10926번, ??!
[prob 10926](https://www.acmicpc.net/problem/10926) : 입출력을 응용하는 문제??! (문자열 입력 뒤에 "??!"을 붙여 출력하기)  
```python
a = input()
print(a + "??!")
```
풀이 : 파이썬에서는 문자끼리 더해서 이어붙일 수 있다.

## 1-9. 18108번, 1998년생인 내가 태국에서는 2541년생?!
[prob 10108](https://www.acmicpc.net/problem/18108) : 태국은 불교국가로, 석가모니 열반한 해를 기준으로 연도를 센다. 불기 연도를 서기 연도로 바꿔 출력하라.  
ex) 불기 2541년 = 서기 1998년
```python
a = int(input())
print(a - (2541 - 1998))
```
풀이 : 불기 연도와 서기 연도 차이만큼 알고싶은 년도에서 빼준다.  

## 1-10. 3003번, 킹, 퀸, 룩, 비숍, 나이트, 폰
[prob 3003](https://www.acmicpc.net/problem/3003) : 흰색 체스 피스는 킹 1개, 퀸 1개, 룩 2개, 비숍 2개, 나이트 2개, 폰 8개로 구성되는데, 입력으로 흰색 킹, 퀸, 룩, 비숍, 나이트, 폰의 개수가 주어진다. 이때 각 입력에 대해 몇 개의 피스를 더하거나 빼야 되는지 출력한다.
```python
# 킹, 퀸, 룩, 비숍, 나이트, 폰 1, 1, 2, 2, 2, 8
a, b, c, d, e, f = map(int, input().split())
print(1 - a, 1 - b, 2 - c, 2 - d, 2 - e, 8 - f)
```
풀이 : 입력값이 한 줄에 여러 개 있고, 모두 정수로 받고 싶을 때 map함수를 활용하면 좋다. map(function, iterable)로 사용가능하다. iterable한 매개변수는 순서가 존재하여 반복가능해야 한다. ex) 리스트, 튜플 등 원래 세트에서 갖고 있는 차이만큼 출력한다.  

## 1-11. 10430번, 나머지
[prob 10430](https://www.acmicpc.net/problem/10430) : (A+B)%C는 ((A%C) + (B%C))%C 와 같은지, (A×B)%C는 ((A%C) × (B%C))%C 와 같은지 알아보기 위해 위의 4가지를 구해본다. 입력은 한 줄에 A, B, C 주어짐.  
```python
a, b, c = map(int, input().split())
print((a + b) % c)
print(((a % c) + (b % c)) % c)
print((a * b) % c)
print(((a % c) * (b % c)) % c)
```
풀이 : 그대로 출력해주면 된다.  
실행 시켜보면 (A+B)%C는 ((A%C) + (B%C))%C 와 같고, (A×B)%C는 ((A%C) × (B%C))%C 와 같음을 알 수 있다.  

## 1-12. 2588번, 곱셈
[prob 2588](https://www.acmicpc.net/problem/2588) : 세 자리수 * 세 자리수를 시행한다. a * (b의 1의 자리), a * (b의 10의 자리), a * (b의 100의 자리), a * b 결과를 차례대로 출력하라.
```python
a = input()
b = input()
a, b = int(a), int(b)
org_b = b
for i in range(3):
    print(a * (b % 10))
    b = int(b / 10)
print(a * org_b)
```
풀이 : b의 자리수를 1의 자리에서 100의 자리로 올려가며 곱해주기 위해 반복문으로 b를 나눠가며 곱해준다.  

## 1-13. 10171번, 고양이
[prob 10171](https://www.acmicpc.net/problem/10171) : 아래처럼 고양이 출력  
![baekjoon 10171](/assets/images/baekjoon_10171.jpg)
{: width="30%" height="30%"}
```python
print("\\    /\\")
print(" )  ( \')")  
print("(  /  )")
print(" \\(__)|")
```
풀이 : 파이썬에서 ', ", \를 문자열로 출력하기 위해선 ', " \ 앞에 \를 붙여준다.  
- 문자열을 " "로 묶은 경우 앞에 \을 붙이지 않더라도 '을 출력할 수 있다. ' '에 대해서도 "를 그냥 출력할 수 있다. 그러나 헷갈리므로 그냥 \을 앞에 붙여 사용하는 게 좋을 것 같다.  

## 1-14. 10172번, 개
[prob 10172](https://www.acmicpc.net/problem/10172) : 아래처럼 개 출력  
![baekjoon 10172](/assets/images/baekjoon_10172.jpg)
{: width="30%" height="30%"}
```python
print("|\\_/|")
print("|q p|   /}")
print("( 0 )\"\"\"\\")
print("|\"^\"`    |")
print("||_/=\\\\__|")
```
풀이 : 고양이 문제와 같다.  

## 1-15. 25083번, 새싹
[prob 25083](https://www.acmicpc.net/problem/25083) : 아래처럼 새싹 출력  
![baekjoon 25083](/assets/images/baekjoon_25083.jpg)
{: width="30%" height="30%"}
```python
print("         ,r\'\"7")
print("r`-_   ,\'  ,/")
print(" \. \". L_r\'")
print("   `~\\/")
print("      |")
print("      |")
```
풀이 : 고양이, 개 문제와 동일

# step1 : 입출력, 사칙연산 요약
너무 기초적인 것이기에 어려운 점은 거의 없었다. 여러 개 변수 한 줄에 받을 때 map(func, iter), input().split() 쓰는 법이랑 ' " \ 출력할 때 앞에 \ 붙여주는 정도만 알면 될 것 같다.  
파이썬은 입력받을 때 변수 타입이 자동으로 정해지기 때문에 연산할 때 변수 데이터 형이 맞는지 고려해주는 것도 필요해 보인다.