import gymnasium as gym
import time
import random
import numpy as np
import pygame
import os
import matplotlib.pyplot as plt
CONSTS = {
    'LEFT': 0,
    'DOWN': 1,
    'RIGHT': 2,
    'UP': 3
}

class HillEnvironment:

    def __init__(self, position=(50, 50)):
        # Set environment variable to position Pygame window
        os.environ['SDL_VIDEO_WINDOW_POS'] = f"{position[0]},{position[1]}"
        self.env = gym.make('FrozenLake-v1', desc=None, map_name='8x8', is_slippery=False, render_mode='ansi')
        self.qTable = np.zeros((64, 4))
        self.current_state, _ = self.env.reset()
        self.statistics = []

    def renderBoard(self):
        self.env = gym.make('FrozenLake-v1', desc=None, map_name='8x8', is_slippery=False, render_mode='human')
        self.current_state, _ = self.env.reset()
        self.env.render()

    def step(self, action: int):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        if not terminated:
            self.current_state = next_state
        return next_state, reward, terminated, truncated, info

    def selectAction(self, epsilon, debug = False):
        if np.random.rand() < epsilon:
            if np.max(self.qTable[self.current_state]) > 0:
                if debug:
                    print(f"Exploiting: choosing best action for state {self.current_state}")
                    print(f"Q-values: {self.qTable[self.current_state]}")
                    print(f"Best action: {np.argmax(self.qTable[self.current_state])}")
                return np.argmax(self.qTable[self.current_state])
        if debug:
            print(f"Exploring: choosing random action for state {self.current_state}")
        return random.randint(0, 3)  # Random action from 0-3
    def visualize_q_table(self):
        """
        Opens a visual grid representation of the Q-table using Pygame.
        Each cell shows the best action and its value for that state.
        """
        # Initialize pygame
        pygame.init()
        
        # Constants for visualization
        GRID_SIZE = 8
        CELL_SIZE = 100  # Increased cell size to prevent overflow
        WINDOW_SIZE = (GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE)
        
        # Colors
        BACKGROUND = (240, 248, 255)  # AliceBlue
        BLACK = (0, 0, 0)
        GRID_LINE = (180, 180, 180)
        ARROW_COLOR = (65, 105, 225)  # RoyalBlue
        TEXT_COLOR = (50, 50, 50)
        HIGHLIGHT = (255, 255, 224)  # LightYellow
        
        # Create window
        screen = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption("Q-Table Visualization")
        
        # Arrow polygon points for different directions
        arrows = {
            3: [(0, -15), (8, 0), (-8, 0)],  # UP
            2: [(15, 0), (0, 8), (0, -8)],   # RIGHT
            1: [(0, 15), (8, 0), (-8, 0)],   # DOWN
            0: [(-15, 0), (0, 8), (0, -8)]   # LEFT
        }
        
        # Fonts for displaying values
        title_font = pygame.font.SysFont('Arial', 14, bold=True)
        value_font = pygame.font.SysFont('Arial', 12)
        
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            screen.fill(BACKGROUND)
            
            # Draw grid
            for i in range(GRID_SIZE):
                for j in range(GRID_SIZE):
                    state = i * GRID_SIZE + j
                    rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    
                    # Highlight cells with significant values
                    best_value = np.max(self.qTable[state])
                    if best_value > 0.1:
                        highlight_rect = pygame.Rect(j * CELL_SIZE + 2, i * CELL_SIZE + 2, 
                                                    CELL_SIZE - 4, CELL_SIZE - 4)
                        pygame.draw.rect(screen, HIGHLIGHT, highlight_rect)
                    
                    # Draw cell border
                    pygame.draw.rect(screen, GRID_LINE, rect, 1)
                    
                    # Find best action for this state
                    best_action = np.argmax(self.qTable[state])
                    best_value = self.qTable[state][best_action]
                    
                    # Draw state number
                    state_text = title_font.render(f"State: {state}", True, TEXT_COLOR)
                    screen.blit(state_text, (j * CELL_SIZE + 10, i * CELL_SIZE + 10))
                    
                    # Draw best value
                    value_text = title_font.render(f"Best: {best_value:.2f}", True, TEXT_COLOR)
                    screen.blit(value_text, (j * CELL_SIZE + 10, i * CELL_SIZE + 30))
                    
                    # Draw arrow for best action if value is significant
                    if best_value > 0.01:
                        # Center of the cell
                        center_x = j * CELL_SIZE + CELL_SIZE // 2
                        center_y = i * CELL_SIZE + CELL_SIZE // 2 + 10  # Offset to center in visible area
                        
                        # Get arrow points for this direction
                        arrow_points = [(center_x + x, center_y + y) for x, y in arrows[best_action]]
                        
                        # Draw the arrow with a thicker outline
                        pygame.draw.polygon(screen, ARROW_COLOR, arrow_points)
                        pygame.draw.polygon(screen, BLACK, arrow_points, 1)
                    
                    # Draw all action values in a compact format
                    action_names = ["L", "D", "R", "U"]
                    for action in range(4):
                        value = self.qTable[state][action]
                        color = TEXT_COLOR
                        if action == best_action and value > 0.01:
                            color = ARROW_COLOR
                        
                        action_text = value_font.render(f"{action_names[action]}: {value:.2f}", True, color)
                        screen.blit(action_text, (j * CELL_SIZE + 10, i * CELL_SIZE + 50 + action * 12))
            
            pygame.display.flip()
            
        pygame.quit()

    def train(self, epochs, stepsPerEpoch, gamma, learningRate, epsilon):
        """
        Train the agent using Q-learning algorithm.
        Google "Q-learning algorithm" to see the equation in its full glory
        ### Args:
            epochs (int): Number of training episodes
            stepsPerEpoch (int): Maximum steps per episode
            gamma (float): Discount factor for future rewards (0-1)
            learningRate (float): Learning rate for Q-value updates (0-1)
            epsilon (float): Exploration rate for epsilon-greedy policy (0-1)
        """
        for i in range(epochs):
            self.current_state, _ = self.env.reset()  # Reset at start of each epoch
            episode_success = 0  # Track success for this episode
            
            for j in range(stepsPerEpoch):
                action = self.selectAction(epsilon)
                tmpState = self.current_state
                
                next_state, reward, terminated, truncated, info = self.step(action)
                
                # Always update Q-value for the action taken
                oldQValue = self.qTable[tmpState, action]
                
                if not terminated:
                    # Non-terminal: include future value
                    nextMax = np.max(self.qTable[next_state])
                    newValue = oldQValue + learningRate * (reward + gamma * nextMax - oldQValue)
                else:
                    # Terminal: no future value (nextMax = 0)
                    newValue = oldQValue + learningRate * (reward - oldQValue)
                    if reward == 1:
                        episode_success = 1  # Mark this episode as successful
                        print("Made it to the goal!")
                    
                self.qTable[tmpState][action] = newValue
                
                if terminated:
                    break  # End episode, outer loop will reset environment
            
            self.statistics.append(episode_success)  # Record episode result
   


# Keep the window open for a few seconds so you can see it
if __name__ == '__main__':
    print("Displaying CliffWalking environment. Press Ctrl+C to exit.")
    env = HillEnvironment()
    env.train(epochs=10000, stepsPerEpoch=10000, gamma=0.90, learningRate=0.90, epsilon=0.8)
    env.renderBoard()
    
    # Simple cumulative success plot
    if env.statistics:
        episodes = list(range(len(env.statistics)))
        cumulative_successes = np.cumsum(env.statistics)
        plt.plot(episodes, cumulative_successes)
        plt.xlabel('Episode')
        plt.ylabel('Cumulative Successes')
        plt.title('Cumulative Training Successes Over Time')
        plt.grid(True)
        plt.show()
        
    for i in range(500):
        action = env.selectAction(1, debug=True)
        next_state, reward, terminated, truncated, info = env.step(action)
