---
title: "'A Deep Reinforcement Learning Framework for Optimal Trade Execution' summary"
excerpt: "'A Deep Reinforcement Learning Framework for Optimal Trade Execution' paper summary"
date: 2023-03-10
categories:
    - CITD 1
tags:
    - RL
    - Cryptocurrency
    - Paper
last_modified_at: 2023-03-15
---
- DRL paper '**A Deep Reinforcement Learning Framework for Optimal Trade Execution**' summary

# How I came to read this paper.  

While working on the project under the theme of '**cryptocurrency execution in CEX using RL**', I was recommended by my mentor of Superpetual to read this paper.

He said that the **overall framework for conducting execution through RL** will be similar to this paper. Although this paper conducts RL in the stock market, there seems to be no problem because the execution itself is not very different from the cryptocurrency market.

# a summary of the paper  

- '**A Deep Reinforcement Learning Framework for Optimal Trade Execution**'  
    (Machine Learning and Knowledge Discovery in Databases. Applied Data Science and Demo Track: European Conference, ECML PKDD 2020, Ghent, Belgium, September 14–18, 2020, Proceedings, Part V Sep 2020 Pages 223–240)  
    link : [DLP for execution](https://link.springer.com/chapter/10.1007/978-3-030-67670-4_14)
    > suggested from a mentor of Superpetual

## **Abstract**
In this article, they proposed **a deep reinforcement learning(DRL) based framework** to learn **to minimize trade execution costs** by **splitting a sell order into child orders** and **execute them sequentially over a fixed period**. 

The framework is based on **a variant of the deep Q-Network(DQN) algorithm** that integrates the **Double DQN**, **Dueling Network**, and **Noisy Nets**.

**the differences from previous research work**, which uses **implementation shortfall as the immediate rewards**  
- uses **a shaped reward structure**   
- incorporate **the zero-ending inventory constraint** into the DQN algorithm by **modifying the Q-func updates** relative to standard Q-learning at the final step
    
<br>

They demonstrate that the DQN based optimal trade execution framework  
- **converges fast** during the training phase  
- **outperforms TWAP, VWAP, AC and 2 DQN algorithms** during the backtesting on 14 US equities  
- **improve the stability** by incorporating the zero ending inventory constraint

## 1. Introduction  
**Algorithmic trading** : computer programs, algorithms, and advanced mathematical models to make trading decisions and transactions in the financial market  
$\rightarrow$ become prevelant in major financial markets since the late 1990s, has dominated the modern electronic trading markets

one of the most crucial problems in the realm of algorithmic trading : **Optimal trading execution**  

> **the profitability of many trading strategies** depends on the **effectiveness of the trade execution**

Optimal trade execution : **how to best trade a set of stock shares at a minimal cost**  
$\Rightarrow$ the trading agent has to **trade off between two scenarios**  
1. **trade quickly** with **the risk of encounting a large amount of loss** due to **limited liquidity with certainty**  
2. **trade slowly** with **the risk of adverse market movement**

- Bertsimas and Lo : pioneers in the realm of optimal trade execution  
$\rightarrow$ They use a dynamic programming approach to find an explicit closed-form solution by minimizing trade execution costs of large transactions over a fixed trading period  
- Huberman, Stanzl and Almgren, Chriss : extend their work  
$\rightarrow$ introducing transaction costs, more complex price impact functions, risk aversion parameters
> **The closed-form analytical solutions**, however, have **strong assumptions** on the underlying **price movement or distributions**

- TWAP, VWAP strategy are prevalent among practioners in financial markets.  
$\rightarrow$ The TWAP, VWAP strategies have very few assumptions
> TWAP and VWAP strategies are not able to learn from historical data.

**RL** enables the trading agent to **interact with the market** and to **learn from its experiences**. **The Q-learning**(a RL technique) does not require **a model of the market** and hence has **fewer assumptions on the price dynamics than the closed-form solutions**. 

- Nevmyvaka, Feng, and Kearns : have published the first large-scale empirical application of reinforcement learning to optimal trade execution problems (2006)   
- Hendricks and Wilcox : propose to combine the Almgren and Chriss model (AC) and RL algorithm and to create a hybrid framework mapping the states to the proportion of the AC-suggested trading volumes

> **The traditional RL techniques** mentioned above are limited by **the curse of dimensionality**. (There are high dimensions and the complexity of the underlying dynamics of the financial market.)

**The Deep Q-Network(DQN)** (combining the deep neural network
and the Q-learning) : can address the curse of dimensionality challenge faced
by Q-learning.  
- known as **the universal function approximator**  
- capable of extracting intricate nonlinear patterns from raw dataset. 

An important **constraint in the execution** problem : all the shares must be executed by the end of the trading period.(**zero-ending inventory constraint**)  
$\Rightarrow$ In the real world business, the brokers receive contracts or directives from their clients to execute **a certain amount of shares** within **a certain time period**.

> **The previous research** are all based on a **generic Q-learning** algorithm which **fails to address the zero-ending inventory constraint**.

**Their main contributions**:  
1. proposing **a modified DQN algorithm** to **address the zero ending inventory constraint**  
2. designing **a shaped reward structure** to standardize the rewards among different stocks and under various market scenarios  
$\rightarrow$ facilitates the convergence of the DQN algorithm and is robust to noise in market data  
3. performing **an extensive experiment** to **demonstrate the advantages of DQN method** over TWAP, VWAP, and AC model as well as Ning et al.’s algorithm and its variant.   
4. carefully **designing the framework** by **integrating OpenAI Gym’s RL framework and Ray’s distributed RLlib**.

**The organization of this paper**: 
- **Section 2** : reviewing the concepts of **limit order book** and **market microstructure**.  
- **Section 3** : revisiting the basics of **DQN algorithm**, presenting our **DQN based optimal trade execution framework** and describing **the experimental methodology and settings**.  
- **Section 4** : describing **the data sources**, discussing and reporting **empirical results**.  
- **Section 5** : **concluding the paper** and proposes
**future work**.

## 2. Limit Order Book and Market Microstructure   
a brief introduction to the **limit order book(LOB)** and **market microstructure**

**LOB** : **records the quantities and the prices** that a buyer or seller is willing to buy or sell for the particular security.

When there is a **cross between bid and ask price** (bid price $\geq$ ask price), the match engine **executes the trade** and remove the crossed bid/ask orders from the order book.

**The bid and ask orders** are **listed on the left and right-hand sides** and are sorted **descending(bid)** and **ascending(ask)** by price, respectively. **The top bid and ask orders** are kown as **best bid and ask**.   
> The difference between the best bid and ask prices is the bid-ask spread adn is closely related to market liquidiry.

- They use NYSE daily millisecond Trades & Quotes (TAQ) data to **reconstruct the limit order book**.  
- After order book reconstruction, they **develop a market environment/simulator** by leveraging OpenAI Gym's framework to **read in the constructed historical order book information sequentially** and **respond to the actions** that RL agent takes based on the rules described above.

> quote : the last price at which an asset traded

## 3. A DQN Formulation to Optimal Trade Execution   
**The nonlinear function approximation** is kown to **cause instability or even divergence in RL**.  
$\rightarrow$ **The DQN algorithm** addresses the instability issue by **experience replay**. The experience replay breaks down the correlations among samples and improves the stability of the DQN algorithm.  
>  **DeepMind** has introduced the **Target Q**, which **freezes the Q functions for a certain amount of steps** instead of updating it in each iteration, to further improve the stability.

### 3.1 Preliminaries  
**The goal of the DQN agent** : to **maximize cumulative discounted rewards** by making **sequential decisions** while **interacting with the environment**. (similar to Q-learning)    

The main difference : DQN is using **a deep neural network** to **approximate the Q function**.

At each time step, we would like to obtain the **optimal Q function**, which obeys the Bellman equation. 

if the **optimal action-value function** $Q^∗(s', a')$ was completely known at next step, the optimal strategy at current step would be to maximize $E[r + γQ^∗(s', a')]$, where γ is the discounted factor 

$$Q^∗(s', a') = E_{s'∼ε}[r + γ \max_{a'} Q^∗(s', a')|s, a]$$

The **Q-learning** algorithm **estimates the action-value function**  
$\rightarrow$ by iteratively updating $Q_{i+1}(s, a) = E_{s'∼ε}[r + γ \max_{a'} Q_i(s', a')|s, a]$.  

It has already been demonstrated that the action-value $Q_i$ would eventually **converge to the optimal action-value** $Q^∗$ as **i → ∞**. The Q-learning iteratively updates the Q table to obtain an optimal Q table.

However, it suffers from **the curse of dimensionality** and is **not scalable to large scale problems**. The DQN algorithm trains **a neural network model to approximate the optimal Q function** by minimizing a sequence of loss function $L_i(θ_i) = E_{s,a∼ρ(·)}[(y_i − Q(s, a; θ_i))^2]$ iteratively,  
where $y_i = E_{s'∼ε}[r + γ \max_{a'} Q^∗(s', a'; θ_{i−1})|s, a]$ is the target function and $ρ(s, a)$ refers to **the probability distribution of states s and actions a**. 

>  In the DQN algorithm, **the target function** is usually **freezed for a while** when optimizing the loss function to **prevent the instability caused by the frequently shift in target function.**

The gradient : 

$$\nabla_{\theta_i} L_i (\theta_i) = E_{s,a∼ρ(·);s'∼ε}[((r + γ \max_{a'} Q^∗(s', a'; \theta_{i−1}))_i − Q(s, a; \theta_i))\nabla_{\theta_i} Q(s, a; \theta_i)]$$

**The model weights** could be estimated by optimizing the loss function through **stochastic gradient descent algorithms(SGD)**.  

**DQN algorithm**  
- capability of **handling high dimensional problems**  
- **model-free RL algorithm** : **no assumption about the dynamics of the environment** (learns about optimal policy by **exploring the state-action space**)

### 3.2 Problem Formulation  
**The Q-learning algorithm** is a **model-free** technique, but **the curse of dimensionality** has limited its application in high dimensional problems.  
$\rightarrow$ **DQN** is capable of **handling high dimensional problems**, **inheriting Q-learning's ability** to learn from its experiences.

In this section, they provide the **DQN formulation** for the optimal
trade execution problem and describe the **state, action, reward, and the algorithm** used in the experiment.

- **States** : a vector to describe the current status of the environment   
  - - -
  1. **Public state** : **market microstructure variables** including top 5 bid/ask prices and associated quantities, bid/ask spread  
  2. **Private state** : **remaining inventory** and **elapsed time**
  3. **Derived state** : they also **derive features based on historical limit order book (LOB) states** to account for the temporal component in the environment    
  $\Rightarrow$ For example, **the differences of immediate rewards** between current time period and arrival time at various volume levels, manually **crafted indicators to flag specific market scenarios** (i.e., regime shift, a significant trend in price changes, and so on)   

  $\rightarrow$ **The derived features**:
    1. **Volatility in VWAP price**   
       $\rightarrow$ use several trading volumes: 0.5*TWAP, 1*TWAP, and 1.5*TWAP to compute the VWAP price   
       (TWAP represents the trading volume of the TWAP strategy in one step.)   
    2. **Percentage of positive change in VWAP price**  
       $\rightarrow$ recod the **steps that the current VWAP price increases compared with the previous step** and compute the percentage of the positive changes
    3. **Trends in VWAP price**  
       $\rightarrow$ calculate **the difference between the current VWAP prices and the average VWAP prices** in the past 6, 12, and 24 steps.

    - derive the features based on the past 6, 12, and 24 steps (each step is a 5-second interval) respectively  


- **Actions**   
  - - -
  They choose **different numbers of shares to trade** based on **the liquidity of the stock market**. 
  
  **The purpose of the research** : to evaluate the **capability** of the proposed framework to **balance the liquidity risk and timing risk**  
  $\rightarrow$ They choose the total number of shares to ensure that **the TWAP orders are able to consume at least the 2nd best bid price on average**.  
  
  ![trading shares](/assets/images/Trading-Shares.png){: width="50%", height="50%" .align-center}

  In the optimal trade execution framework, they set the range of actions **from 0 to 2TWAP** and discretize the action space into 20 equally distributed grids. Hence, we have **21 available actions** including: **0, 0.1TWAP, 0.2TWAP, ..., 1.9TWAP, 2TWAP**. The learned policy maps the state to the 21 actions.

  > TWAP order = $\frac{Total \, \sharp \, of \, shares \, to \, trade}{Total \, \sharp \, of \, periods}$

- **Rewards**  
  - - -
  The **reward structure** is the key to the DQN algorithm and should be carefully designed to **reflect the goal of the DQN agent**. Otherwise, the DQN agent cannot learn the optimal policy as expected.

  In contrast to previous work, we use a **shaped reward structure** rather than the **implementation shortfall**(a commonly used metric to measure execution gain/loss) as the immediate reward received after execution at each non-terminal step.

  > **Implementation Shortfall** = arrival price × traded volume - executed price × traded volume  
  
  > **arrival price** = the mid-point between the bid/ask prices at the time the order was placed  

  > **mid price** = the average price between a seller's ask price of a stock and the best buyer bid price of that stock  
  (**quoted price** : the last price at which an asset traded)

  There is a **common misconception** that the **implementation shortfall (IS) is a direct optimization goal**.  
  $\rightarrow$ The **IS** compares **the model performance** to **an idealized policy** that assumes **infinite liquidity at the arrival price**. 
  
  In the real world, brokers often use **TWAP and VWAP as benchmarks**. IS is usually used as the optimization goal  
  $\rightarrow$ because **TWAP and VWAP prices** could **only be computed until the end of the trading horizon** and **cannot be used for real-time optimization**.  
  
  Essentially, **IS** is just **a surrogate reward function**, but **not a direct optimization goal**. Hence, a reward structure is good as long as it can improve model performance.

  Additionally, there are some shortcomings of IS reward:  
    1. it’s **noisy** making learning hard  
    2. it’s **not stationary** across scenarios:  
      1. **price is going up**:  
      All actions might receive positive rewards while some actions might be bad decisions.  
      2. **price is going down**:  
      All actions might receive negative rewards while we may still want the agent to trade constantly or even more aggressively to prevent an even larger loss at the end.

  **The shaped reward structure** attempts to **remove the impact of trends** to **standardize the reward signals** and is demonstrated as follows. 
  
  It is worth mentioning that **the shaped reward structure** below is **only roughly tuned on Facebook’s training data**. Although **it might not be optimal for all the stocks**, it generalizes reasonably well on the rest equities in the article as demonstrated in Sect. 4.

  ![structured reward](/assets/images/structured-reward.png){: width="70%", height="70%" .align-center}

  ![structured reward2](/assets/images/structured-reward2.png){: width="70%", height="70%" .align-center}

  

- **Zero Ending Inventory Constraint**  
  - - -
  The major difference in this approach compared with previous research is that this approach **combines the last two steps (time T-1 and T)** and **estimate the Q-function for them together**.

    
- **Assumptions**  
  - - -
  - The actions that DQN agent take have **only a temporary market impact**  
  - The market is resilient and will **bounce back to the equilibrium level** at the next time step.  
  
  > The DQN agent's actions do not affect the behaviors of other market participants.  
  > As the equities we choose are liquid, and the **actions are relatively small** compared with the market volumes

  -  **ignore the commissions and exchange fees**  
  
  > As this research is primarily aimed at institutional investors, and those fees are relatively small fractions and are negligible.

  - apply a **quadratic penalty** if the **trading volume exceeds the available volumes**    
  - The **remaining unexecuted volumes will be liquidated all at once at the last time step** to ensure the execution of all the shares   
  - assume direct and fast access to exchanges with **no order arrival delays**  
  - if **multiple actions** result in the **same reward**, we **choose the maximum action (trading volumes)**

   

### 3.3 DQN Architecture and Extensions   
**The vanilla DQN algorithm** has addressed the **instability issues** of using the nonlinear function approximation by techniques (**experience replay**, **the target network**)

In this article, we also incorporate several other techniques to improve its **stability** and **performance** further.

- **Network Architecture**  
  - - -
  - **fully connected feedforward neural** network with **2 hidden layers**,  
    **128 hidden nodes** in each hidden layer, **ReLU** activation function  
  - **input layer** : **51 nodes** (private attributes (remaining inventory, time elapsed), derived attributes)  
  - output layer : 21 nodes, linear activation function  
  - **Adam optimizer**

- **Double DQN**   
  - - -
  It is well known that the **maximization step** tends to **overestimate the Q function**.

  - The **Double DQN** could **reduce the harmful overestimation** and **improve the performance of DQN**.

  The only changes is in the loss function below.  

  $$ \nabla_{\theta_i} L_i (\theta_i) = E_{s, a \sim p(\cdot);s' \sim \varepsilon} [((r + \gamma Q(s', max_{a'} Q^* (s', a'; \theta_{i-1})))_i - Q(s, a; \theta_i))\nabla_{\theta_i} Q(s, a; \theta_i)] $$


- **Dueling Networks**  
  - - -
  The dueling network architecture consists of **two streams of computations**   
  - one stream representing **state values**  
  - the other one representing **action advantages**. 
  
  The two streams are combined by an aggregator to output an estimate of the Q function. The dueling architecture **learn the state-value function more efficiently**.


- **Noisy Nets**  
  - - -
  **Noisy nets** are proposed to **address the limitations of the $\epsilon$-greedy exploration policy** in vanilla DQN. 
  
  A **linear layer** with a **deterministic and noisy stream** replaces the standard linear $y = b + W x$. 
  
  The noisy net enables the network to **learn to ignore the noisy stream** and **enhance the effectiveness of exploration**.

  $$ y = (b + Wx) + (b_{noisy} \odot \epsilon^b + (W_{noisy} \odot \epsilon^w)x) $$


> I will skip the parts from 3.4 ~ 4.3. It will be better to read the article itself.  

- 3 : A DQN Formulation to Optimal Trade Execution
  - 3.4 : Experimental Methodology and Settings  
- 4 : Experimental Results   
  - 4.1 : Data Sources   
  - 4.2 : Training and Stability    
  - 4.3 : Main Evaluation and Backtesting 

## 5. Conclusion and Future Work   
In this article, we propose an **adaptive trade execution framework** based on a **modified DQN algorithm** which **incorporates the zero-ending inventory constraint**. 

The experiment results suggest that the **DQN algorithm is a promising technique for the optimal trade execution** and has demonstrated **significant advantages** over TWAP, VWAP, AC model as well as Ning et al.’s algorithm and its variant. 

We use **shaped rewards as the reward signals** to encourage the DQN
agent to conduct correct behaviors and has demonstrated its **advantage over the commonly used implementation shortfall**.

The current experiment relies on the **assumption** that **the DQN agent’s actions are independent of other market participants’ actions**. 

It will be exciting if we could **model the interactions of multiple DQN agents and their collective decisions in the market**.

