---
title: "밑바닥부터 시작하는 딥러닝 ch.3"
excerpt: "ch.3 신경망"
date: 2022-11-23
categories:
    - AI
tags:
    - DL
use_math: true
last_modified_at: 2022-11-23
---

<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }

    table.dataframe td {
      text-align: center;
      padding: 8px;
    }

    table.dataframe tr:hover {
      background: #b8d1f3; 
    }

    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  </style>
</head>


# 3. 신경망

- 퍼셉트론들을 연결해 신경망을 구성할 수 있다.


## 3.1 퍼셉트론에서 신경망으로

신경망은 여러 층으로 이뤄져있는데 제일 왼쪽(낮은)층은 **입력층**, 중간 층은 **은닉층**, 맨 오른쪽(마지막 층)은 **출력층**이라고 한다. 입력층이나 출력층과 달리 은닉층은 개발자 외에는 보이지 않는다. 



이전에는 퍼셉트론에서 입력 노드를 x1, x2로만 뒀는데 편향이 추가된 후로는 x1, x2, 1으로 노드를 구성하고 1에 해당하는 가중치를 b로 둔다.  

- 식 1 $$ y = \begin{cases}0 \quad (b + w_1x_1+w_2x_2 \leq 0) \\ 1 \quad (b + w_1x_1+w_2x_2 \gt 0) \end{cases} $$

위의 식을 아래처럼 나타낼 수 있다.  
- 식 2 $$ y = h(b + w_1x_1 + w_2x_2) $$
- 식 3 $$ h(x) = \begin{cases} 0 \quad (x \leq 0) \\ 1 \quad (x \gt 0) \end{cases} $$



위에서 h(x)와 같이 입력 신호의 총합을 출력 신호로 변환해주는 함수를 **활성화 함수 activation function**이라고 한다. 입력 신호의 총합이 활성화를 일으키는지 정하기 때문에 activation function이다. '식 2'를 다시 써보면 아래와 같다.

$$ a = b + w_1x_1 + w_2x_2 $$

$$ y = h(a) $$

이처럼 일반적으로 나타내는 뉴런은 

1. '가중치\*입력 + 편향'의 총합을 계산
2. 합을 활성화 함수에 대입해 출력  



두가지 단계로 구성된다.

> 일반적으로 **단순 퍼셉트론**은 단층 네트워크에서 계단 함수를 activation function으로 사용한 모델을 가리키고, **다층 퍼셉트론**은 여러 층으로 구성되고 시그모이드 함수 등의 매끈한 활성화 함수를 사용하는 네트워크를 가리킨다.


## 3.2 활성화 함수

위에서 살펴본 h(x) 같이 임계값을 경계로 상수 결과값이 바뀌는 함수를 **계단 함수 step function**이라고 한다. 퍼셉트론 activation function으로 사용 가능한 여러 후보 중 계단함수를 사용하는 것이다. 이제부터는 계단 함수 외에 다른 activation function을 알아본다.



- 시그모이드 함수 sigmoid function  

$$ h(x) = \frac 1 {1 + \exp(-x)} $$

sigmoid function은 신경망에 자주 활용되는 활성화함수이다. $\exp(-x)$는 $e^{-x}$를 뜻하고 $e$는 자연상수로 2.7182~ 의 값을 갖는 상수다. sigmoid function의 치역은 0~1이고, x값이 커질수록 1에 가까워지고 x값이 작아질수록 0에 가까워진다.

***

계단 함수를 그래프를 통해 확인해보면 다음과 같다.

```python
import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    return np.array(x > 0, dtype = np.int) 
# x > 0으로 boolean 배열을 만들고, dtype으로 np.int를 통해 boolean 배열을 0, 1 값으로 변환한다.

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)    # y축 범위 설정
plt.show()
```

<img src="data:image/png;base64,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"/>

***

마찬가지로 sigmoid function도 그래프를 그려보면 아래와 같다.



```python
import numpy as np
import matplotlib.pylab as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1,1)
plt.show()
```

<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAXQAAAD8CAYAAABn919SAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAf80lEQVR4nO3deXRV5b3/8feXzAkhJBDGMISZIIMQQbB1qBNQK97aQcWx7aVqtfa2tmLrrVp/va231+letZRa9WKrVIsDtTjhbetIJcyEeTQhQAYyz8l5fn8kdUUMcAjnZJ/h81orK9ln7ySfszj5rIfn7P1sc84hIiLhr4fXAUREJDBU6CIiEUKFLiISIVToIiIRQoUuIhIhVOgiIhHihIVuZk+aWbGZbT7GfjOz/zazXWa20cymBj6miIiciD8j9KeB2cfZPwcY3f6xAPj1qccSEZGTdcJCd869Axw5ziHzgCWuzSqgt5kNDFRAERHxT2wAfsZgoKDDdmH7YwePPtDMFtA2iiclJWXauHHjAvDrRUSix5o1a0qdc5md7QtEoVsnj3W6noBzbjGwGCA3N9fl5eUF4NeLiHSvVp+jsr6ZiromyuuaqapvprL9o6q+merGFqobmqlqaKGmoYWaxhZqG9s+1zW1ct3M4dx2wegu/W4z23+sfYEo9EJgSIftLKAoAD9XRKTbtPocpTWNHKpsoLi6keLqBkqqGymtaaS0uomy2kbKaps4UttEZX0zx1sGKzGuB6mJcaQmxpKaEEtKQix9UpLpmRBLckIMOYN6BeU5BKLQlwO3mNlSYAZQ6Zz7zHSLiIiXmlp8HKio5+MjdRQcqaOwvJ6iinoOVNRzsKKew9WNtPo+29LpyXH07ZlAn57xjB/Qi4yUeNJT4klPjiM9OZ605Dh6J8WR1v6RmhhHfKw3Z4SfsNDN7DngXKCvmRUCdwNxAM65RcAKYC6wC6gDbghWWBGR43HOUVrTxM7ianYX17C7pJbdJTXsK6vlQHk9Hfs6LsYYmJbE4N5JnDmyDwPTEhmQlsSAXon0S02gX68E+vZMIC4mfC7XOWGhO+euPMF+B3wnYIlERPzQ3Opj5+Ea8osq2XKwim0Hq9l+uJojtU2fHJMSH0N2ZgqTs3pz2ZTBDM1Ibvvok0z/1ER69OjsLcDwFYgpFxGRoHLOUVhez5r95awvqGBDYQX5RVU0tfgASIqLYeyAVC7K6c+Y/qmM6Z/KqH496d8rAbPIKu3jUaGLSMhxzrGruIYP95Sxak8ZefvKKa5uBNrKe2JWGtfNHMZpg9OYMCiN7L4pxETYaLsrVOgiEhJKaxp5d2cJf99ewnu7yiitaSvwQWmJzBzZh9xh6UwblsGY/j2JDaN57e6kQhcRTzjn2HaompVbDrNyWzEbCioA6JMSz+dG92XWyD7MHNGXIRlJUTVtcipU6CLSbZxz5BdV8ZdNB1mx6SD7y+oAmDKkN7dfNIZzx/YjZ2CviHuzsruo0EUk6AqO1PHSugO8vO4Ae0prielhzBrZhxvPGcn54/vRLzXR64gRQYUuIkHR0NzKG/mHeO6jj1m1p219vzNHZLDg7BFcPGEA6SnxHieMPCp0EQmogiN1LPlwHy+sKaSirpmhGcncftEYLjt9MFnpyV7Hi2gqdBE5Zc45Ptp7hN+9t5eVWw9jZsyeMIArpw9l1sg+mhPvJip0Eekyn8/xf9uKefxvu1j7cQXpyXHcdO5Irj5zGAPTkryOF3VU6CJy0pxzvJF/mIdX7mDboWqy0pO4b94Evpo7hMS4GK/jRS0Vuoj4zTnH37aX8MBb29l8oIoRfVN46OuT+dKkQbrYJwSo0EXEL5sPVPIfK7bywe4yhmYk88BXJzNvioo8lKjQReS4iqsbuP+17by4rpDeSXHce+kErpoxNKyWlY0WKnQR6VRLq48lH+7nobd20Nji49tnj+Tm80bSKzHO62hyDCp0EfmMDQUV3LFsI9sOVXP2mEzuvXQC2X1TvI4lJ6BCF5FP1De18uBb2/nde3vJTE1g0dVTuXjCAC2OFSZU6CICwJr95fzg+fXsK6vjqhlDWThnnKZXwowKXSTKNbf6+J+3d/LoX3cxqHcSz/7rDGaN7Ot1LOkCFbpIFPu4rI5bl65jQ0EFl0/N4p5Lc0jVqDxsqdBFotTrmw/xwz9twIDH509l7sSBXkeSU6RCF4kyTS0+fvnaNp58fy+Ts9J49KqpDMnQKoiRQIUuEkVKaxq5+fdr+WjfEa6fNZwfzx1PfKwuEIoUKnSRKLH5QCULluRRVtvEI1dMYd6UwV5HkgBToYtEgRWbDvL959eTkRzPsptmcdrgNK8jSRCo0EUimHOO3767h/9YsY1pw9L5zTXT6NszwetYEiQqdJEI1dLq494/b+GZVfv54sSBPPC1yVqrPMKp0EUiUENzK7ctXccb+Yf59tkjuGP2ON0GLgqo0EUiTE1jCwuW5PHB7jLu/lION5yV7XUk6SYqdJEIUl7bxPVPfcTmoioe/Npkvjw1y+tI0o1U6CIRoqymkflP/IM9pbX85uppXJDT3+tI0s38uqLAzGab2XYz22VmCzvZn2ZmfzazDWaWb2Y3BD6qiBxLaU0jV/32H+wtreXJ685QmUepExa6mcUAjwFzgBzgSjPLOeqw7wBbnHOTgXOBB8wsPsBZRaQTJdWNXLl4FfuP1PLU9WfwudFaKTFa+TNCnw7scs7tcc41AUuBeUcd44BUa1sFvydwBGgJaFIR+Yzy2iaufuIfFJbX89T105k1SmUezfwp9MFAQYftwvbHOnoUGA8UAZuA25xzvqN/kJktMLM8M8srKSnpYmQRAahuaOa6pz5ib1ktT1yXy8yRfbyOJB7zp9A7O3nVHbV9MbAeGARMAR41s16f+SbnFjvncp1zuZmZmScdVkTa1DW18I2nV7OlqIpfz5/KWRqZC/4VeiEwpMN2Fm0j8Y5uAF50bXYBe4FxgYkoIh01t/q46fdrWbO/nIe+PoXzx+sNUGnjT6GvBkabWXb7G51XAMuPOuZj4HwAM+sPjAX2BDKoiLStzXLHso38fUcJP/+XiXxp8iCvI0kIOeF56M65FjO7BXgDiAGedM7lm9mN7fsXAfcBT5vZJtqmaO5wzpUGMbdIVLr/9e28uPYA379wDFdOH+p1HAkxfl1Y5JxbAaw46rFFHb4uAi4KbDQR6ejp9/ey6O+7mT9jKLd+YZTXcSQE6VYlImFg5ZbD3PvqFi7M6c/P5p1G2xnCIp+mQhcJcflFlXx36TomDk7jv684nRitmijHoEIXCWGHqxr45tN5pCXF8cS1uSTFaz1zOTYVukiIamhu5Vv/m0d1QzO/u+4M+vVK9DqShDittigSgpxzLFy2kc1FlSy+JpecQZ+5Tk/kMzRCFwlBT7y7l5fXF/H9C8ZwoVZOFD+p0EVCzDs7SvjFa1uZO3EAt+j0RDkJKnSREFJwpI5bn1vHmP6p/Oork3V6opwUFbpIiGhobuXmP6zF5xy/uWYaKQl6i0tOjl4xIiHi3j/ns+lAJb+9NpdhfVK8jiNhSCN0kRDwfF4Bz31UwM3njtSboNJlKnQRj20/VM2/v7yZmSP68P0Lx3gdR8KYCl3EQ3VNLXzn2bWkJsbxyJVTiI3Rn6R0nebQRTx09yv57C6p4ZlvzKBfqq4ElVOj4YCIR15aV8gLawr5zrmj+Nxo3UJOTp0KXcQD+0prueulzZwxPJ3vXTDa6zgSIVToIt2sudXHbUvXEdPDeOSK0zVvLgGjOXSRbvbwyh1sKKzksaumMqh3ktdxJIJoaCDSjVbtKePxv+3mq9Oy+OKkgV7HkQijQhfpJpX1zXz/j+sZlpHMPZdO8DqORCBNuYh0k3uW53O4upFlN83SOi0SFBqhi3SDFZsO8tK6A9xy3iimDOntdRyJUCp0kSArrmrgJy9tYlJWmtY3l6BSoYsEkXOOO5ZtpK6plQe/NoU4naIoQaRXl0gQPZ9XwF+3l7BwzjhG9evpdRyJcCp0kSA5UFHPfa9u5cwRGVw3c7jXcSQKqNBFgsA5x8JlG/E5x6++MpkePXQrOQk+FbpIEDz3UQHv7izlx3PHMyQj2es4EiVU6CIBVlhex8//soWzRvVh/oyhXseRKKJCFwkg5xx3vrgJgPsvn4SZplqk+/hV6GY228y2m9kuM1t4jGPONbP1ZpZvZn8PbEyR8PDCmkLe3VnKwjnjyErXVIt0rxNef2xmMcBjwIVAIbDazJY757Z0OKY38Dgw2zn3sZn1C1ZgkVB1uKqB+17dwvTsDObPGOZ1HIlC/ozQpwO7nHN7nHNNwFJg3lHHXAW86Jz7GMA5VxzYmCKhzTnHXS9vpqnFx/2XT9JZLeIJfwp9MFDQYbuw/bGOxgDpZvY3M1tjZtd29oPMbIGZ5ZlZXklJSdcSi4SgVzce5K0th/nBRWPI7pvidRyJUv4UemdDDXfUdiwwDfgicDHw72Y25jPf5Nxi51yucy43MzPzpMOKhKLy2ibuWZ7PpKw0vnFWttdxJIr5s4ZnITCkw3YWUNTJMaXOuVqg1szeASYDOwKSUiSE/XzFVirrm3nmmzN0OznxlD+vvtXAaDPLNrN44Apg+VHHvAJ83sxizSwZmAFsDWxUkdDz3s5S/rSmkG+fM4KcQb28jiNR7oQjdOdci5ndArwBxABPOufyzezG9v2LnHNbzex1YCPgA55wzm0OZnARr9U1tXDnSxsZ0TeFW78w2us4Iv7dscg5twJYcdRji47a/hXwq8BFEwltD6/cScGRepYuOJPEuBiv44joSlGRrth8oJLfvbeXK6cP4cwRfbyOIwKo0EVOWquv7fL+9OR4Fs4e73UckU+o0EVO0tMf7GPTgUru/lIOaclxXscR+YQKXeQkHKio54E3t3Pe2EwumTTQ6zgin6JCF/GTc46fvrwZ5+Bn807TSooSclToIn56ffMh3t5WzPcvHKObVkhIUqGL+KGqoZm7l+eTM7AXN5w13Os4Ip3y6zx0kWj3wBvbKalp5LfX5uryfglZemWKnMD6ggqWrNrPdTOHM3lIb6/jiByTCl3kOFpafdz54ib6pSbwg4s+s4CoSEjRlIvIcTz1/j62Hqxi0dVTSU3UOecS2jRCFzmGAxX1PPjWDs4f14+LJwzwOo7ICanQRTrhnOPuV9oWDL133gSdcy5hQYUu0ok38g+zcmsx/3bhaLLSdc65hAcVushRahpbuGd5PuMGpHKDbiknYURviooc5YE3t3O4uoHHr55KnM45lzCiV6tIB5sPVPK/H+xj/oyhTB2a7nUckZOiQhdp9891zvv0TOCHF4/zOo7ISVOhi7Rb8mHbOuc/vSSHtCSdcy7hR4UuAhysrOeBN3dw9hitcy7hS4UuAtyzPJ8Wn4+fX6Z1ziV8qdAl6r2Zf4g38g9z2/la51zCmwpdolpNYwt3L89nbP9UvvV5nXMu4U3noUtUe+itHRysbODRq3TOuYQ/vYIlam0qrOSp9/cyf8ZQpg3TOecS/lToEpVaWn0sfHEjfXsm8KPZOudcIoOmXCQqPfX+PvKLqvj1/Kk651wihkboEnUKjtTx4Fs7uGB8f2afpnXOJXKo0CWqOOe46+XN9DD4mdY5lwijQpeo8sr6Iv6+o4QfXjyWQb2TvI4jElAqdIkaZTWN3PvnfE4f2ptrZg73Oo5IwPlV6GY228y2m9kuM1t4nOPOMLNWM/tK4CKKBMZ9r26hprGF+y+fREwPTbVI5DlhoZtZDPAYMAfIAa40s5xjHHc/8EagQ4qcqr9uL+bl9UXcfO4oxvRP9TqOSFD4M0KfDuxyzu1xzjUBS4F5nRx3K7AMKA5gPpFTVtPYwl0vbWZUv57cfN5Ir+OIBI0/hT4YKOiwXdj+2CfMbDDwL8Ci4/0gM1tgZnlmlldSUnKyWUW65D9f30ZRZT33Xz6RhNgYr+OIBI0/hd7ZZKM7avth4A7nXOvxfpBzbrFzLtc5l5uZmelvRpEu+8eeMpZ8uJ8bZmUzbViG13FEgsqfK0ULgSEdtrOAoqOOyQWWtp/T2xeYa2YtzrmXA5JSpAvqm1q5Y9lGhmYkc/vFY7yOIxJ0/hT6amC0mWUDB4ArgKs6HuCc+2TdUTN7GnhVZS5ee2jlDvaV1fHst2aQHK9VLiTynfBV7pxrMbNbaDt7JQZ40jmXb2Y3tu8/7ry5iBfWflzOE+/u4crpQ5k1qq/XcUS6hV/DFufcCmDFUY91WuTOuetPPZZI1zU0t3L7CxsYmJbEj+dqJUWJHvp/qEScB97czp6SWn7/zRmkJmolRYkeuvRfIkreviM88V7bTSs+N1pTLRJdVOgSMeqbWvnhnzYyKC2JO+eO9zqOSLfTlItEjF+8tpW9pbU8+60Z9EzQS1uij0boEhHe2VHCkg/3842zsnVWi0QtFbqEvcq6Zn70p42M6teTH80e63UcEc+o0CXs/XT5ZkprGnnoa1NIjNNaLRK9VOgS1l5Zf4BX1hfx3fNHMzErzes4Ip5SoUvYKjhSx10vbWbasHRuPlfL4oqo0CUstbT6+Lc/rgfg4a9PITZGL2URndslYenxv+0mb385D399CkMykr2OIxISNKyRsLNm/xEeeXsnl00ZxGWnDz7xN4hECRW6hJWKuiZufXYdg3sn8bPLTvM6jkhI0ZSLhA3nHLe/sJGSmkaW3TSLXlp4S+RTNEKXsPHU+/tYufUwd84Zz6Ss3l7HEQk5KnQJCxsKKvjFa1u5YHx/bjhruNdxREKSCl1CXnltEzf/YS39UhP5r69Oov3etSJyFM2hS0hr9Tlu++N6Sqob+dNNM+mdHO91JJGQpUKXkPY//7eTd3aU8B//MlHz5iInoCkXCVl/3V7MI2/v5PKpWVw5fYjXcURCngpdQtKekhq++9w6xg/oxf+77DTNm4v4QYUuIae6oZl/XZJHXEwPFl87jaR4LYkr4g8VuoQUn8/xvaXr2V9Wx+Pzp5KVrnVaRPylQpeQ8qs3t/P2tmLu/lIOZ47o43UckbCiQpeQ8fzqAn79t91cNWMoV585zOs4ImFHhS4h4YPdpfz4pU18fnRf7r10gt4EFekCFbp4bldxDTc+s4bsvik8Nn8qcbpZhUiX6C9HPHW4qoHrnvyIuJgePHn9GVpBUeQU6EpR8UxVQzPXP7Wa8romli44U3ceEjlFGqGLJxpbWvn2kjXsPFzNr6+epsv6RQLAr0I3s9lmtt3MdpnZwk72zzezje0fH5jZ5MBHlUjxzxs8f7injP/8yiTOGZPpdSSRiHDCQjezGOAxYA6QA1xpZjlHHbYXOMc5Nwm4D1gc6KASGXw+xx3LNrFi0yHu+uJ4vjw1y+tIIhHDnxH6dGCXc26Pc64JWArM63iAc+4D51x5++YqQH+l8hnOOe5ens+ytYX82wVj+NbnR3gdSSSi+FPog4GCDtuF7Y8dyzeB1zrbYWYLzCzPzPJKSkr8TylhzznHL17bxjOr9vPts0fw3fNHeR1JJOL4U+idXeHhOj3Q7DzaCv2OzvY75xY753Kdc7mZmZo3jRbOOX7+l60sfmcP184cxsI543ThkEgQ+HPaYiHQcTHqLKDo6IPMbBLwBDDHOVcWmHgS7pxz3PfqVp58fy/XzxrO3V/KUZmLBIk/I/TVwGgzyzazeOAKYHnHA8xsKPAicI1zbkfgY0o48vkc9yzP58n39/KNs7JV5iJBdsIRunOuxcxuAd4AYoAnnXP5ZnZj+/5FwE+BPsDj7X+wLc653ODFllDX3OrjR3/ayEvrDrDg7BHcqWkWkaAz5zqdDg+63Nxcl5eX58nvluBqaG7llmfXsnJrMT+8eCw3nztSZS4SIGa25lgDZl36LwFVUdfEgiVrWL3/CPdddhrXaBlckW6jQpeA+bisjuuf/ojCI/U8csXpXDp5kNeRRKKKCl0CYn1BBd98ejUtPsfvvzWD6dkZXkcSiToqdDllL60rZOGyTfTvlchTN5zByMyeXkcSiUoqdOmyVp/j/te3sfidPczIzuDx+VPp0zPB61giUUuFLl1SVtPI9/64nnd3lnLdzGHcdUmO7jQk4jEVupy01fuOcOuz6zhS18T9l0/k62cM9TqSiKBCl5Pg8zl+884e/uvN7QxJT+LFm2Zx2uA0r2OJSDsVuvilqKKeHzy/gQ/3lDF34gB+efkk3f9TJMSo0OW4nHMs31DEv7+8mRaf4/7LJ/K13CG68lMkBKnQ5ZiKqxq46+XNvLnlMFOH9uahr09hWJ8Ur2OJyDGo0OUznHO8kFfIfX/ZQlOLjzvnjOObn8smVmexiIQ0Fbp8ytaDVfz0lc2s3lfO9OwMfvnliYzQhUIiYUGFLgBU1jXz8Ns7WPLhftKS4rj/8ol8ddoQevTQXLlIuFChR7mmFh/PrNrPf7+9k6qGZubPGMrtF42ld3K819FE5CSp0KNUq8/x6sYiHnxrB/vL6vj86L7cOWc8OYN6eR1NRLpIhR5lfD7H6/mHeOitHewsrmHcgFSevuEMzhmTqVMRRcKcCj1KNLf6WL6+iEV/383O4hpG9evJo1edztzTBmqeXCRCqNAjXGV9My/kFfDU+/s4UFHPuAGpPHLFFC6ZNIgYFblIRFGhR6gdh6t55sP9LFtbSF1TK9OHZ3DfZRM4b2w/Ta2IRCgVegSpb2rlL5sO8txHH7NmfznxMT24dMogrp81XItoiUQBFXqYa/U5Vu0p48W1B3h980Fqm1oZ0TeFn8wdz5enDtYNJ0SiiAo9DLX6HHn7jvCXTQd5bfMhSqobSU2I5ZJJg/jy1MFMz87QtIpIFFKhh4m6phbe21nKyq2H+b9txZTWNJEQ24PzxvbjkskDuWB8fxLjYryOKSIeUqGHKJ/Pse1QNe/uLOGdnSWs3ltOU6uP1MRYzh3bj4ty+vOFcf1ISdA/oYi0URuEiJZWH9sOVbNmfzkf7i7jH3vLKK9rBmBM/55cO3MY543rx/TsDN27U0Q6pUL3SHFVA+sLKthQWMGGgkrWfVxObVMrAIN7J3H++P6cOaIPZ43qw8C0JI/Tikg4UKEHWXOrjz0ltWw7VMX2Q9XkF1WRX1RFaU0jADE9jLH9U/ny1Cxyh6czbVg6WenJHqcWkXCkQg8A5xxHapvYV1bLvtI69pbWsqu4hl0lNewvq6W51QEQ28MY1a8n54zJZMKgXkweksaEQWl6M1NEAkKF7odWn6O0ppGDlQ0cqqynsLyeoooGCsvrKCivp+BIHTWNLZ8cH9PDGNYnmZGZPbkwpz9j+6cydkAqIzJTSIhVeYtIcERtoTe1+Kiob6K8tpmy2sZPPpdWN1JS00RJdSMl1Q0crmqkpKaRVp/71Pcnx8cwuHcSQzOSmZGdwZCMZLL7JjO8TwpDMpL1xqWIdDu/Ct3MZgOPADHAE865Xx6139r3zwXqgOudc2sDnBVoO52vvrmVuqZW6ppaqGtqpbaxhZrGFmob276ubmyhuqGZ6oa2z1X1LVTWN3/qo+OIuqMeBhkp8fTtmUC/XomM7p9K/14JDEhLYmCvRAakJZKVnkRaUpwu3hGRkHLCQjezGOAx4EKgEFhtZsudc1s6HDYHGN3+MQP4dfvngHt100G++9w6v45NiY+hV1IcqYmxpCXFMTAtkXEDUklLjiM9OZ705DjSU+LJSImnT0oCGe1faxVCEQlH/ozQpwO7nHN7AMxsKTAP6Fjo84AlzjkHrDKz3mY20Dl3MNCBJwzqxZ1zxpGcEEtyXAzJ8TGkJMSSkhBLz4RYeia2fU6Jj9Fd6kUkqvhT6IOBgg7bhXx29N3ZMYOBTxW6mS0AFgAMHTr0ZLMCMDKzJyPP0V3oRUSO5s8QtrP5B9eFY3DOLXbO5TrncjMzM/3JJyIifvKn0AuBIR22s4CiLhwjIiJB5E+hrwZGm1m2mcUDVwDLjzpmOXCttTkTqAzG/LmIiBzbCefQnXMtZnYL8AZtpy0+6ZzLN7Mb2/cvAlbQdsriLtpOW7wheJFFRKQzfp2H7pxbQVtpd3xsUYevHfCdwEYTEZGTofP6REQihApdRCRCqNBFRCKECl1EJEKo0EVEIoQKXUQkQqjQRUQihApdRCRCqNBFRCKECl1EJEKo0EVEIoQKXUQkQqjQRUQihApdRCRCWNvKtx78YrMSYL8nv/zU9AVKvQ7hgWh83tH4nCE6n3c4PedhzrlO7+HpWaGHKzPLc87lep2ju0Xj847G5wzR+bwj5TlrykVEJEKo0EVEIoQK/eQt9jqAR6LxeUfjc4bofN4R8Zw1hy4iEiE0QhcRiRAqdBGRCKFCPwVmdruZOTPr63WWYDOzX5nZNjPbaGYvmVlvrzMFk5nNNrPtZrbLzBZ6nSfYzGyImf3VzLaaWb6Z3eZ1pu5iZjFmts7MXvU6y6lSoXeRmQ0BLgQ+9jpLN3kLOM05NwnYAdzpcZ6gMbMY4DFgDpADXGlmOd6mCroW4AfOufHAmcB3ouA5/9NtwFavQwSCCr3rHgJ+BETFu8rOuTedcy3tm6uALC/zBNl0YJdzbo9zrglYCszzOFNQOecOOufWtn9dTVvBDfY2VfCZWRbwReAJr7MEggq9C8zsUuCAc26D11k88g3gNa9DBNFgoKDDdiFRUG7/ZGbDgdOBf3ibpFs8TNvAzOd1kECI9TpAqDKzlcCATnb9BPgxcFH3Jgq+4z1n59wr7cf8hLb/nv+hO7N1M+vksaj4n5iZ9QSWAd9zzlV5nSeYzOwSoNg5t8bMzvU6TyCo0I/BOXdBZ4+b2UQgG9hgZtA29bDWzKY75w51Y8SAO9Zz/iczuw64BDjfRfYFDIXAkA7bWUCRR1m6jZnF0Vbmf3DOveh1nm5wFnCpmc0FEoFeZvZ759zVHufqMl1YdIrMbB+Q65wLl5XausTMZgMPAuc450q8zhNMZhZL2xu/5wMHgNXAVc65fE+DBZG1jU7+FzjinPue13m6W/sI/Xbn3CVeZzkVmkMXfz0KpAJvmdl6M1vkdaBgaX/z9xbgDdreHHw+ksu83VnANcAX2v9917ePXCWMaIQuIhIhNEIXEYkQKnQRkQihQhcRiRAqdBGRCKFCFxGJECp0EZEIoUIXEYkQ/x8COj2IwTjQPAAAAABJRU5ErkJggg=="/>

sigmoid를 정의할 때 1 / (1 + np.exp(-x) 부분에서 x가 np.array()가 대입되어도 문제없다. 위에서 배웠던 numpy의 브로드캐스팅 덕분에 연산이 각 원소별로 이뤄지기 때문이다.

> sigmoid는 's자 모양'이라는 뜻이다. step function 처럼 함수 모양을 본따 sigmoid function이라고 이름 붙여졌다.

***

sigmoid function과 step function의 차이점  

- sigmoid function은 부드러운 곡선이고, 출력이 연속적으로 변한다.(미분 가능하다.)
- 반면 계단함수는 0을 경계로 출력값이 0과 1로 나뉘고, 0을 제외한 지점에서 미분계수가 0이다. 또한 0에서 미분 불가능하다.

> step function을 사용하는 경우 뉴런 사이에 0 혹은 1이 흐르는 반면, sigmoid function을 사용한 경우 연속적인 실수가 흐른다.

***

sigmoid function과 step function의 공통점

- 큰 관점에서 비슷한 모양이다. 입력이 작아질수록 0에 가깝고, 커질수록 1에 가깝다.
- 출력이 0 ~ 1이다.
- 둘 다 비선형 함수이다.

***

step function은 구부러진 직선이기 때문에 비선형 함수이다.  

선형 함수는 수식 $f(x) = ax + b$로 표현되는 함수로 직선 1개로 나타난다. **비선형 함수 non-linear function**는 선형 함수가 아닌 함수를 뜻한다. 즉, 직선 1개로 나타낼 수 없는 함수를 뜻한다.



신경망에서는 activation function으로 non-linear function을 사용해야 한다. linear function을 사용하면 신경망의 층을 깊게 하는 의미가 사라진다. 간단히 예를 들면 $h(x) = cx$를 activation function으로 갖는 3층 네트워크를 생각하면 $y(x) = h(h(h(x)))$가 된다. 이는 $y = c^3*x$ 이고 은닉층이 없는 네트워크로 표현가능하다. 즉 non-linear function 없이 층을 쌓는 것은 단순히 선형결합이 반복되기 때문에 하나의 함수로 근사하는 것과 유사한 결과밖에 내지 못한다는 것이다. 이는 결국 layer를 쌓는 이점을 잃게된다.



Sigmoid function은 신경망에서 유용하게 활용되었으나 몇몇의 문제점들로 최근에는 ReLU function을 주로 활용한다. Sigmoid function의 문제점은 나중에 알아보도록 하고 ReLU function을 알아본다.  

***

- ReLU function (Rectified Linear Unit)  

rectify는 수정하다, 바로잡다 등의 뜻을 가진다. ReLU는 입력이 0이하면 0을 출력하고, 0보다 크면 그대로 입력을 출력하는 함수다. 

$$ h(x) = \begin{cases} x \quad (x \gt 0) \\ 0 \quad (x \leq 0) \end{cases} $$

ReLU function의 그래프는 아래와 같다.



```python
import numpy as np
import matplotlib.pylab as plt

def relu(x):
    return np.maximum(0, x)

x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)
plt.plot(x, y)
plt.show()
```

<img src="data:image/png;base64,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Rw2gyX72txItWL9Tv72Wctpk4EQNoEn+umCjHn1rjW44u6dG9qecNhkIagAxW1G2R3dQTpt0BDWAmOwvr9KEGUVq1eIITRmTq5ZZxEeysKMGEJPvPbNIy0p369GvDaWcNsl4SATQqDnvrtMThet14/m9dO5pFIMkG0EN4FO9v7GmnPbsUztp8gW9wh4nIxHUAA5r94EKTZxZpI5tWuqBUYMopw0JO2oADapfTjt73HB1akc5bVg4UQNo0O/fqCmnvf3i3jozh3LaMBHUAD6haO12/fz5El3Yp7PGnUs5bdgIagAfsX1vuSblF9WW0w6kBCAC2FEDOKSunHbrnnI9Of4symkjghM1gEOmzVuueUvL9L0r+uqMbkeHPQ5qEdQAJNWU0943d5muHHiirh3WI+xxUA9BDeBQOW1PymkjiR01kOEqq6o1aVax9h6s1MxvDFNbymkjh78RIMPdO3eZ/lVbTntaZ8ppo4jVB5DBXi7ZoofmrdDooT0op40wghrIUOu27dO35yxQvxM76O4r+oY9Dj4FQQ1koIOVVZo0s0jVTjltKmBHDWSgnx4qpx2ik46jnDbqOFEDGeavCzbqj2+t0dfP7qmR/U8IexzEgKAGMkhdOe2Qk47RdyinTRmNBrWZdTezV82sxMwWm9nkZAwGILH2lVdq/IxCHdkyS1PGDKacNoXEsqOulHSLuxeZWXtJhWY2193fD3g2AAni7rrrz4v0QekePfq1oepyNOW0qaTRh1R33+TuRbUf75ZUIqlr0IMBSJzH3l2np4o2UE6bopr0bx8zy5E0WNI7DXxunJkVmFlBWVlZYqYDELfFG3fq+39ZTDltCos5qM2snaQnJd3k7rs+/nl3n+7uee6el53NIzYQBbsOVGhCfpGOoZw2pcX0PGoza6makM5396eCHQlAIri7bn98odZv3085bYqL5VkfJulhSSXufl/wIwFIhEfeWK2/L96s74yknDbVxbL6GCHpOknnm9n82h+XBjwXgDgUrtmunz9Xos/37axvnEM5baprdPXh7q9LYrEFpIhte8s1aWaRunRsrXsop00LvNcHkEaqq103PTZfH9aV0x5FOW064KVJQBqZ+upyvbasTN+nnDatENRAmnhz+Vbd/9IyXTXoRI2lnDatENRAGtiy64C+NbtYJ2e308++QDltumFHDaS4yqpq3TirWHsP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"/>

> ReLU 에서 Rectified는 '정류된'이란 뜻으로, 정류는 전기회로 쪽 용어로 ReLU가 0 이하일 때를 차단하여 0을 출력하는 것과 유사하여 ReLU라고 이름지었다.


## 3.3 다차원 배열 계산

numpy의 다차원 배열을 이용한 계산법을 익히면 신경망을 효율적으로 구현 가능하다.



숫자가 한줄, 직사각형, 3차원 N차원으로 나열된 것들을 통틀어 다차원 배열이라고 한다. 1차원부터 numpy로 배열을 작성해본다.



```python
import numpy as np
a = np.array([1, 2, 3, 4])
print(a)
np.ndim(a)       # 배열의 차원 수
print(a.shape)   # 인스턴트 변수 shape가 tuple 형식으로 배열 형태를 저장한다.
print(a.shape[0])
```

<pre>
[1 2 3 4]
(4,)
4
</pre>

***

2차원 배열을 numpy로 구현하면 아래와 같다.
```python
b = np.array([[1, 2], [3, 4], [5, 6]])
print(b)
print(np.ndim(b))
print(b.shape)
```
<pre>
[[1 2]
 [3 4]
 [5 6]]
2
(3, 2)
</pre>

***

2차원 배열은 특히 **행렬 matrix**라고 부른다. 가로 방향을 행 row, 세로 방향을 열 column이라고 부른다.

행렬(2차원 배열)의 내적(행렬 곱)을 구하는 방법을 알아본다.  

행렬 A와 행렬 B의 내적은 A행렬의 행(가로)와 B의 열(세로)을 원소별로 곱해서 더한 것이 새로운 다차원 배열의 원소가 된다. 

- AB와 BA는 결과가 다를 수 있음에 주의한다.  
- AB를 할 때 A의 열 수와 B의 행 수가 같아야 한다. ex) (3 * 2) * (2 * 4)

np.dot(a, b)를 통해 행렬 곱을 계산 가능하다. a와 b는 numpy array이어야 한다.


## 3.4 3층 신경망 구현하기

3층 신경망에서의 입력부터 출력까지의 처리(순방향 처리)를 구현한다.

입력층 2개, 1번째 은닉층 3개, 2번째 은닉층 2개, 출력층 2개의 뉴런으로 구성된다.

(X : 입력, W : 가중치, B : 편향, A : 가중 신호와 편향의 총합, Z : activation function 결과값)


(identity function은 굳이 정의할 필요 없지만, 흐름을 통일하기 위해서 사용했다.)  

> 출력층의 activation function은 문제의 성질에 맞게 달리한다. 회귀 regression 에서는 identity function을, 2클래스 분류에는 sigmoid를, 다중 클래스 분류는 softmax를 사용하는 것이 일반적이다.



```python
def identity_function(x):
    return x

# parameter 값 초기화
def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    
    return network

# 순전파 진행
def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)
    
    return y
    
network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)   
```

<pre>
[0.31682708 0.69627909]
</pre>

***

(신경망 수현 관례에 따라 가중치만 대문자로 작성하고, 나머지는 소문자로 작성하였다고 한다.) init_network()로 parameter 값을 초기화해준 후에 forward 함수를 정의해주면, 순전파 진행을 y = forward(network, x)를 통해 한번에 처리 가능하다.


## 3.5 출력층 설계하기

신경망은 **분류 classification**, **회귀 regression** 같은 다루는 문제들에 따라 출력층의 activation function이 달라진다. 일반적으로 regression은 항등 함수를, classification은 softmax function을 활용한다.

> classification은 데이터가 어느 클래스에 속하는지 알아내는 문제이고, regression은 입력 데이터에서 연속적인 수치를 예측하는 문제이다.

>> 분류와 달리 **회귀**는 이름이 직관적이지 않다. '회귀'의 유래는 사람과 완두콩의 키(크기)를 부모 자식간에 예측하는 데서 유래했다. 키가 큰 부모의 자식은 작고, 키가 작은 부모의 자식은 컸다는 점에서 평균으로 회귀한다 것을 알았다. 부모와 자식 키 사이엔 선형 관계가 있어 부모 키로부터 자식키를 예측할 수 있고, 예측 결과값이 연속적인 수치인 것이 회귀 문제가 뜻하는 개념이 되었다.



regression에 사용하는 항등함수는 입력을 그대로 출력한다. classification에서 사용하는 softmax function은 다음과 같다.

$$ y_k = \frac {\exp(a_k)}{\displaystyle\sum_{i = 1}^{n}\exp(a_i)}$$

$\exp(x)$는 $e^x$(지수함수)를, $n$은 출력층의 뉴런 수, $y_k$는 그 중 k번째 출력, $a_k$는 k번째 입력 신호를 뜻한다. softmax function은 다음과 같이 코드로 정의가능하다.



```python
def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    
    return y
```

softmax function을 구현할 때는 overflow 문제를 주의해야 한다. exponential function을 다루기 때문에 값이 너무 커지면 계산이 어려워진다. 이를 해결하려면 softmax function에 다음과 같이 조작을 가해야 한다.



$$ \begin{aligned}

y_k = \frac {\exp(a_k)}{\displaystyle\sum_{i = 1}^{n}\exp(a_i)} 

&= \frac {C\exp(a_k)}{C\displaystyle\sum_{i = 1}^{n}\exp(a_i)} \\

&= \frac {\exp(a_k + \log C)}{\displaystyle\sum_{i = 1}^{n}\exp(a_i + \log C)} \\

&= \frac {\exp(a_k + C')}{\displaystyle\sum_{i = 1}^{n}\exp(a_i + C')}

\end{aligned}$$



0이 아닌 임의의 정수 C를 분모 분자에 곱해줘서 조작을 가한다. 마지막 줄을 보면, exponential 안에 $\log C$가 더해진 것을 알 수 있다. 즉, softmax의 exponential을 계산할 때, 어떤 정수를 더하거나 빼도 결과는 동일하다는 것이다. $C'$에 어느 값을 대입하던 상관  없지만, overflow를 막는 목적으로 입력신호 중 최댓값을 빼주는 것이 일반적이다.



```python
a = np.array([1010, 1000, 990])
print(np.exp(a) / np.sum(np.exp(a)))   # 처리 전 softmax, nan : not a number

c = np.max(a)
a -= c
print(np.exp(a) / np.sum(np.exp(a)))
```

<pre>
[nan nan nan]
[9.99954600e-01 4.53978686e-05 2.06106005e-09]
</pre>

***

위처럼 softmax를 다음과 같이 다시 구현해주면 된다.
```python
def softmax(a):
    c = np.max(a)
    a -= c
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    
    return y
```

softmax function의 중요한 특징이 있다.  

바로 softmax 값이 0 ~ 1 사이의 실수라는 것이다. 또한, softmax 출력의 총합은 1이다.  어떤 실수들이 0~1 사이의 실수값을 가지고 총합이 1이면 그것은 확률로 해석 가능하다. 여기서 왜 softmax가 classification에 활용되는지 눈치챌 수 있다. k번째 출력이 k번째 class일 확률이다. 



그러나 softmax를 적용해도 exponential은 단조 증가 함수이기 때문에 원소의 대소값이 변하지 않는다. 즉, 분류에서는 가장 큰 출력을 내는 뉴런이 무엇인지만 알면 되기 때문에 신경망으로 분류할 때는 출력층의 softmax를 생략해도 된다. 현업에서도 exponential의 계산에 드는 자원 낭비를 줄이고자 출력층의 softmax를 생략한다고 한다. (exponential은 계산하는 cost가 크다.)

> 기계학습은 학습과 추론 두가지 단계를 거치는데, 학습이 끝나고 추론을 할 때는 softmax를 생략하지만, 학습 단계에서는 softmax를 사용한다고 한다. (4장 참조)



출력층의 뉴런 수는 풀려는 문제에 맞게 적절하게 정한다. 분류에서는 분류하고 싶은 클래스 수로 설정하는 것이 일반적이다. 


## 3.6 손글씨 숫자 인식

신경망의 실전 예시를 적용해본다. 이번에는 학습 완료된 parameter를 가지고 추론 과정만 구현한다. 추론 과정을 신경망의 **순전파 forward propagation**라고도 한다.  



MNIST는 손글씨 숫자 이미지 집합으로, 기계학습 분야에서 유명한 dataset이다. 0부터 9까지의 숫자 이미지로, 훈련 이미지가 60,000장, 시험 이미지가 10,000장 준비되어 있다. 이미지는 28 \* 28 크기의 회색조 이미지이며, 각 픽셀은 0~255 값을 취한다. 또한 각 이미지는 '7', '2' 같이 실제 의미하는 숫자 label이 붙어있다.



책에서 제공하는 코드에서는 dataset 파일을 불러오기 위한 경로 지정으로 상위폴더를 불러오기 위해 os.pardir를 사용해서 불러오는데, 오류가 발생한다. 하위폴더로 codes를 넣어준다.



```python
import sys, os
from codes.dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

print(x_train.shape)  # (60000, 784)
print(t_train.shape)  # (60000,)
print(x_test.shape)   # (10000, 784)
print(t_test.shape)   # (10000,)
```

<pre>
(60000, 784)
(60000,)
(10000, 784)
(10000,)
</pre>

***

MNIST 데이터를 확인할 겸 이미지를 화면에 출력해본다. Python Image Library PIL 모듈로 이미지를 표시한다.



dataset.mnist에 접근해서 load_mnist를 import한다. 이후, 데이터를 (훈련 이미지, 훈련 레이블), (시험 이미지, 시험 레이블) 형태로 받아온다. 
- flatten : (True:784개 원소인 1차원 배열로 이미지 저장), (False:1 \* 28 \* 28의 3차원 배열로 저장)
- normalize : (True:이미지 픽셀 값 0.0\~1.0 사이 값으로 정규화), (False:0\~255 값)  
- one_hot_label : (True:정답 뜻하는 원소만 1 \[0,0,1,0,0,0,0,0,0,0\]), (False:숫자 형태로 label 저장 '7')  

```python
# coding: utf-8
import sys, os
import numpy as np
from codes.dataset.mnist import load_mnist
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))    # numpy image 데이터 PIL용 데이터 객체로 변환 
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = t_train[0]
print(label)  # 5

print(img.shape)  # (784,)  # Flatten = True 이기 때문이다
img = img.reshape(28, 28)  # 형상을 원래 이미지의 크기로 변형
print(img.shape)  # (28, 28)

img_show(img)
```

<pre>
5
(784,)
(28, 28)
</pre>

***

> 파이썬에는 pickle 이라는 기능으로, 프로그램 실행 중 특정 객체를 파일로 저장하는 기능이다. MNIST 데이터셋을 읽는 load_mnist() 함수도 2번째 읽기부터 pickle을 이용해 바로 데이터를 불러온다.



신경망 추론 처리 : neuralnet_mnist.py



```python
import sys, os
import numpy as np
import pickle
from codes.dataset.mnist import load_mnist
from codes.common.functions import sigmoid, softmax

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

x, t = get_data()
network = init_network()
accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p= np.argmax(y) # 확률이 가장 높은 원소의 인덱스를 얻는다.
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
```

<pre>
Accuracy:0.9352
</pre>

***

입력층은 28\*28 = 784 개이고, 출력층은 0\~9 = 10개이다. 첫 은닉층은 50개, 두번째 은닉층은 100개 뉴런을 가진다. 은닉층은 임의로 정한 값이다. 추론 과정만 구현하기 때문에 init_network()에서 초기화된 매개변수를 불러온다. 또한 load_mnist()에서 normalize = True를 했다. 이처럼 데이터를 특정 범위로 변환하는 처리를 **전처리 pre-processing**이라고 한다. (입력 이미지 데이터에 대한 전처리 작업으로 정규화 normalizing 을 수행한 셈이다.)



> 현업에서도 딥러닝에 전처리를 활발히 사용한다. 전처리를 통해 식별 능력 개선, 학습 속도 향상이 가능하다. 데이터 전체의 분포를 고려해 전처리 하는 경우가 많다. 데이터 전체 평균, 표준편차를 이용해 데이터들이 0 중심으로 분포하도록 이동, 데이터의 확산 범위를 제한한다. 그외에 전체 데이터를 균일하게 분포시키는 데이터 **백색화 whitening**도 있다.



이미지 100개를 묶어 predict() 함수에 넘기는 경우, 형상은 다음과 같다.  

X : 100 * 784, W1 : 784 * 50, W2 : 50 * 100, W3 : 100 * 10, Y : 100 * 10  

출력 데이터는 100 * 10이 된다. 100장의 입력 데이터의 결과 (10개의 클래스에 대한 확률)이 나타난다. 이처럼 하나로 묶는 입력 데이터를 **배치 batch**라고 한다.

> batch 처리를 함으로써 큰 배열로 이뤄진 계산을 하는데, 큰 배열을 한꺼번에 계산하는 게 분할된 작은 배열 여러 번 계산하는 것보다 빠르다.



신경망을 batch 처리 해서 다시 구현하여 본다.



```python
x, t = get_data()
network = init_network()

batch_size = 100 # 배치 크기
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
```

<pre>
Accuracy:0.9352
</pre>

***

argmax()를 수행할 때, axis = 1은 각 행별로 최댓값 index를 반환한다. axis = 0은 각 열별로 최댓값 index를 반환한다.

## 3.7 정리
- 신경망에선 activation function으로 sigmoid, ReLU 같은 매끄럽게 변화하는 함수를 사용한다.
- 기계학습에는 regression, classification 등이 있다.
- regression 에서는 identity, classification 에서는 softmax를 사용한다.
- classification 출력층 뉴런 수는 분류 클래스 수와 같게 한다.
- 입력 데이터를 묶은 것을 batch라 하며, batch 단위로 진행할 시 결과를 빠르게 얻을 수 있다.
