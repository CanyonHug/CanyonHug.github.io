---
title: "Reinforcement Learning taxonomy"
excerpt: "RL taxonomy"
date: 2023-03-20
categories:
    - CITD 1
    - AI
tags:
    - RL
last_modified_at: 2023-03-26
---
- Taxonomy of RL algorithms


# 1. Basic concepts of RL

![RL icon](/assets/images/RL.png){: width="30%" height="30%" .align-center}

**Reinforcement Learning (RL)** : a area of '**machine learning**' concerned with how **intelligent agents** ought to take **actions** in **an environment** in order to **maximize the notion of cumulative reward**

![RL 2 icon](/assets/images/RL-2.png){: width="30%" height="30%" .align-center}


- **State** : a vector to describe the current status of the environment

- **Policy** : a rule used by an agent to decide what actions to take. It can be deterministic or stochastic.  

  - **Deterministic policy** : one action for a given state,  
  
  $$ a_t = \mu(s_t) $$
  
  - **Stochastic policy** : a probability distribution of actions for a given state  
  
  $$ a_t \sim \pi(\cdot|s_t) $$
  
  In deep RL, we deal with **parameterized policies**. (policies whose outputs are **computable functions** that depend on **a set of parameters** (eg the weights and biases of a neural network) which we can adjust to change the behavior via some optimization algorithm.)

  We often denote the parameters of such a policy by $\theta$ or $\phi$, and then write this as a subscript on the policy symbol to highlight the connection:  

  $$ a_t = \mu_\theta(s_t) $$

  $$ a_t \sim \pi_\theta(\cdot|s_t) $$

- **Reward** : the reward given by the environment as a kind of feedback  

  The reward function $R$ depends on the current state or state-action pair.  
  - **Reward functions** that depends on the **current state** 

  $$ r_t = R(s_t) $$

  - **Reward function** that depends on the **state-action pair**  

  $$ r_t = R(s_t, a_t) $$ 

  The goal of the agent is to maximize some notion of **cumulative reward over a trajectory**, which is called '**return**'.

  - **finite-horizon undiscounted return**  

    $$R(\tau)=\sum_{t=0}^T r_t$$

  - **infinite-horizon discounted return**  

    $$R(\tau)=\sum_{t=0}^\infty \gamma^t r_t$$
  
    ($\gamma$ is a discount factor $\gamma \in (0, 1)$. It gives penalty for the reward obtained in the future.)


- **Value functions** : the value of a state, or state-action pair.  

  It means the **expected return** if you start in that state or state-action pair, and then act according to a particular policy forever after.

  There are four main functions. (from OpenAI Spinning Up)

  ![RL Value functions](/assets/images/RL-value_funcs.png){: width="70%" height="70%" .align-center}

> Open AI spinning up RL informations ([RL algorithms](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html))


# 2. Taxonomy of RL algorithms

![RL components](/assets/images/RL-components.png){: width="40%" height="40%" .align-center}

There are some criteria for RL algorithms  
- Model-Based vs Model-Free  
- Policy-Based vs Value-Based  
- On-policy vs Off-Policy

## Model?

Model : The agent’s expectation of what the next state and reward of the environment will be.  

- **State model**   

$$P_{ss'}^a = P[S_{t+1} = s'|S_t=s, A_t = a] $$

- **Reward model**   

$$R_{s}^a = P[R_{t+1} |S_t=s, A_t = a] $$

There are pros and cons of having the model.  
- **pros** : As agent know how environment will change depending the agent’s action, the agent can act more efficiently.  
- **cons** : As a ground-truth model of the environment is usually not available to the agent, it could lead to the error of the agent.


## Model-Based vs Model-Free  

- **Model-based** : method of using a model  
$\rightarrow$ '**Learn the model**' or '**Given the model**'

- **Model-Free** : method of not using a model  
$\rightarrow$ '**Policy optimization**' or '**Q-Learning**'

> In many cases, the model-free method is used a lot because it is difficult to obtain a model in real world.


## Policy-Based vs Value-Based  

- **Policy-based** : explicitly build a representation of a policy  
$\rightarrow$ Policy optimization

> Policy-based agents have the advantage of **learning more reliably** because they are directly optimized for what they want.

- **Value-based** : store only a value function(or action-value function(=Q function)).   
$\rightarrow$ The policy is implicit, pick the action with the best value.   
$\rightarrow$ Q-Learning

> Value-based agents have the advantage of **using data more efficiently**.

- **Actor-critic** : mix of Policy-based and Value-based


## On-policy vs Off-policy

- **On-policy** : for policy updating, **use only data obtained from the most current policy** 
  - (usually for policy-based, policy optimization)

- **Off-policy** : for policy updating, use any data  
  (**data should not be obtained from the most current policy.**)  
  - (usually for value-based, Q-learning)


## Policy Optimization vs Q-Learning

There are 2 main learning method on Model-Free RL.

- **Policy Optimization** : The family of 'Policy Optimization' has **explicit policy** $\pi_\theta (a|s)$. They optimize parameter $\theta$ by using Gradient Ascent method for objective function $J(\pi_{\theta})$.  
  These optimizations are usually done **on-policy**.

- **Q-learning** : The family of 'Q-learning' learn an **action-value function** $Q_\theta (s, a)$ that approximates the optimal action-value function $Q^* (s, a)$. In general, they use a target function based on **the Bellman equation**. (usually done in **off-policy**) 

  It uses the following implicit policy.  

  $$ a(s) = arg \, \max_{a} Q_\theta (s, a) $$


## RL taxonomy tree

  ![RL Taxonomy tree](/assets/images/RL-taxonomy-tree.png){: width="70%" height="70%" .align-center}