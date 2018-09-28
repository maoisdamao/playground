import numpy as np
import pandas as pd
from maze_env import Maze


class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions       # should be a list
        self.lr = learning_rate      # alpha
        self.gamma = reward_decay    # gamma
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
    
    def choose_action(self, observation):
        self.check_state_exists(observation)
        if np.random.uniform() < self.epsilon:
            state_action = self.q_table.loc[observation, :]
            # there could be same value among actions, random choose one if then
            action = np.random.choice(state_action[state_action==np.max(state_action)].index) 
        else:
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        # update = self.lr * (q_target - q_predict)
        self.check_state_exists(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        else:
            q_target = r
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)
    
    def check_state_exists(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state
                )
            )


def update():
    for episode in range(100):
        observation = env.reset()

        while True:
            env.render()
            action = RL.choose_action(str(observation))
            observation_, reward, done = env.step(action)
            RL.learn(str(observation), action, reward, str(observation_))
            observation = observation_

            if done:
                break
    print("Game Over")
    env.destroy()


if __name__ == '__main__':
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()