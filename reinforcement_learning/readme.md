# Learning reinforcement learning with practices
This collection includes practices for [morvan's RL tutorial](https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/2-1-general-rl/)


## Practice 1. Treasure on the right
A practice for Q-learning.

In Q-learning, we maintains a Q-table with q-values for each (state, action) pair. Our explorer take action according to the q-table and policy defined.

#### Game Definition
![image for example 1](https://github.com/Pennsy/blogmao/blob/gh-pages/img/find_the_treasure.gif)

'T' on the right of the line stands for the treasure, and 'o' represents our observer. The goal of our observer 'o' is to find the treasure 'T' in less steps.
#### Reward Rule
The reward rule for the game is simple, 1 for reach the treasure and 0 for all other states.
#### Action Strategy
We follow $\epsilon$ -greedy to take actions, where $\epsilon$=0.9. So, under 90%, observer will choose best action with higher q-value, under 10%, it will take action randomly.
#### Update Policy
The policy we are following to update Q-table,

$$Q(s,a)\rightarrow (1-\alpha)Q(s,a) + \alpha[r+\gamma max_{a'} Q(s', a')]$$
,where $\alpha$ is the learning rate, and $\gamma$ is the discount rate.


