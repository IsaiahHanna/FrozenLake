import time
import random
import numpy as np
import pandas as pd
import gym

class Q_Learning():
    def __init__(self,env,seed = 42,eps=0.1,gamma= 0.95,stepsize = 0.05):
        self.env = env
        self.seed = seed
        self.eps = eps
        self.gamma = gamma
        self.stepsize = stepsize
        self.Q = np.zeros((env.observation_space.n,env.action_space.n))
        np.random.seed(seed)
        random.seed(seed)
    
    def greedy(self,s):
        '''
        Greedy policy in regards to Q

        Parameters:
        Q: Numpy matrix containing all state action values
        s: state index

        Returns:
        The index of the action that with the maximum state action value given the current state
        '''
        return np.argmax(self.Q[s])

    def eps_greedy(self,s):
        '''
        Epsilon greedy policy in regards to Q

        Parameters:
        Q: Numpy matrix containing all state action values
        s: state index
        eps: The epsilon value (The percent chance of choosing a random action)

        Returns:
        The index of the action chosen
        '''
        if np.random.random() < self.eps:
            return np.random.randint(self.Q.shape[1])
        else:
            return self.greedy(s)
    
    def episode(self,env = None):
        '''
        Implementation of Q-Learning algorithm based on pseudo code from Chapter 6 of "Reinforcement Learning: An Introduction"
        
        Returns:
        time_steps: The number of time steps the agent took to complete the episode
        '''
        if env == None:
            env = self.env
        time_steps = 0 
        s,_ = env.reset(seed = self.seed)
        terminated = False

        action = self.eps_greedy(s)
        while not terminated:
            s_prime,reward,terminated,_,_ = env.step(action)

            action_prime = self.eps_greedy(s_prime)
            self.Q[s,action] = self.Q[s,action] + self.stepsize * (reward + (self.gamma * self.Q[s_prime,np.argmax(self.Q[s_prime])]) - self.Q[s,action])
            s = s_prime
            action = action_prime
            time_steps += 1

        return time_steps,reward

    def train(self,num_episodes:int):
        '''
        Function to run through 'num_episodes' episodes and watch for how optimized the agent's policy becomes over time

        Parameters:
        num_episodes: The number of episodes to train over

        Returns:
        time: pd.Dataframe containing the number of steps each episode took
        '''
        results = pd.DataFrame({
            "episodes": [num for num in range(num_episodes)],
            "time_steps" : [0 for time in range(num_episodes)],
            "reward" : [0 for reward in range(num_episodes)]
        })
        for ep in range(num_episodes):
            steps,reward = self.episode()
            results.at[ep,"time_steps"] = steps
            results.at[ep,"reward"] = reward

        return results
    
    def watch(self,env_name,**kwargs):
        '''
        Function for the user to watch the agent go through the environment with the trained policy

        Parameters:
        env_name: The name of the gym environment
        kwargs: Additional arguments needed for gym.make()

        Returns:
        The episode's final reward
        '''
        watch_env = gym.make(env_name,**kwargs,render_mode = "human")
        
        s,_ = watch_env.reset(seed = self.seed)
        terminated = False
        while not terminated:
            action = self.greedy(s) # Note that we use greedy() instead of eps_greedy() as we no longer want to explore.
            s,reward,terminated,_,_ = watch_env.step(action)
            time.sleep(0.3)
        
        watch_env.close()
        return reward



if __name__ == "__main__":
    env = gym.make("FrozenLake-v1",is_slippery = True)
    q_learning = Q_Learning(env)
    q_learning.train(5000)
    env.close()
    q_learning.watch("FrozenLake-v1",is_slippery = True)
