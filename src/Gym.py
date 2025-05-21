import gymnasium as gym
import time
import random
import numpy as np
import pygame
import os
CONSTS = {
    'UP': 0,
    'DOWN': 2,
    'LEFT': 3,
    'RIGHT': 1
}

class HillEnvironment:

    def __init__(self, position=(50, 50)):
        # Set environment variable to position Pygame window
        os.environ['SDL_VIDEO_WINDOW_POS'] = f"{position[0]},{position[1]}"
        
        self.env = gym.make('FrozenLake-v1', desc=None, map_name='8x8', is_slippery=True, render_mode='ansi')
        self.qTable = None
        self.current_statte, _ = self.env.reset()

    def renderBoard(self):
        self.env.reset()
        self.env.render()

    def step(self, direction: int):
        # Fix the double step issue
        return self.env.step(direction)

    def initialiseState(self):
        self.qTable = np.zeros((64, 4))

    def selectAction(self, epsilon):
        if np.random.rand() < epsilon:
            # Exploit: choose action with highest value
            if np.max(self.qTable[self.current_statte]) > 0:  # Check if max value is not zero
                return np.argmax(self.qTable[self.current_statte])
        return random.randint(0, 3)  # Random action from 0-3

    def train(self):
        
#Steps in reinforcement learning

""" 
1. Make policy?
2. Agent moves around
2. Create Q table
3. Update Q table
4. Converge?




 """



# Keep the window open for a few seconds so you can see it
if __name__ == '__main__':
    print("Displaying CliffWalking environment. Press Ctrl+C to exit.")
    env = HillEnvironment()
    env.initialiseState()
    for i in range(10000):
        for j in range(1000):
            action = env.selectAction(0.9)
            next_state, reward, terminated, truncated, info = env.step(action)
            if(terminated == True):
                if reward == 1:
                    print("Made it to the end!")
                env.env.reset()
                break  # This only breaks the inner loop


   