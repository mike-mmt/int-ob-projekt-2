import random
import math
import numpy as np
from typing import List
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import copy

# Constants for the environment
MIN_X = 15.0
MAX_X = 485.0
MIN_Y = 15.0
MAX_Y = 485.0
CENTER_X = 250.0
CENTER_Y = 250.0
TANK_SIZE = 10.0

DISPLAY = False
GRID_SIZE = 10  # Discretization for coordinates
ANGLE_BIN_SIZE = 10  # Discretization for angle difference
ALPHA = 0.1  # Learning rate
GAMMA = 0.9  # Discount factor
EPSILON = 0.1  # Exploration rate

Q = np.zeros((47, 47, 36, 3))

def random_01():
    return random.randint(0, 1)

def calculate_angle(x1, y1, x2, y2):
    return (360 + math.atan2(y2 - y1, x2 - x1) * 180 / math.pi) % 360

def calculate_angle_difference(desired_angle, tank_rotation):
    return ((desired_angle - tank_rotation) + 180) % 360 - 180

class QLearningAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = np.zeros(state_size + [action_size])
    
    def get_state(self, x, y, angle_diff):
        x_discrete = min(int((x - MIN_X) / GRID_SIZE), (MAX_X - MIN_X) // GRID_SIZE - 1)
        y_discrete = min(int((y - MIN_Y) / GRID_SIZE), (MAX_Y - MIN_Y) // GRID_SIZE - 1)
        angle_discrete = int(angle_diff // ANGLE_BIN_SIZE) + 18  # Shift to 0-based index for angle
        return (x_discrete, y_discrete, angle_discrete)
    
    def choose_action(self, state):
        if random.uniform(0, 1) < EPSILON:
            return random.choice(range(self.action_size))
        return np.argmax(self.q_table[state])
    
    def learn(self, state, action, reward, next_state):
        predict = self.q_table[state][action]
        target = reward + GAMMA * np.max(self.q_table[next_state])
        self.q_table[state][action] += ALPHA * (target - predict)
    
    def save_Q(self, filename):
        np.save(filename, self.q_table)

class Stage:
    def __init__(self):
        self.tanks: List[Tank] = []
        self.bullets: List[Bullet] = []
        self.running = True
        if DISPLAY:
            self.fig, self.ax = plt.subplots()
            self.ax.set_xlim(0, 500)
            self.ax.set_ylim(0, 500)
            self.ax.set_aspect('equal')
            self.fig.show()
            self.fig.canvas.draw()
    
    def end_game(self):
        self.running = False
    
    def tick(self):
        for i in range(len(self.tanks)):
            if i == 0:
                self.tanks[i].tick()
                self.tanks[i].cannon.tick()
            if i == 1:
                self.tanks[i].perform_action(6)
                self.tanks[i].cannon.tick()
        # for tank in self.tanks:
            # tank.tick()
            # tank.cannon.tick()
            if DISPLAY:
                self.tanks[i].update_visual()
        for bullet in self.bullets:
            bullet.move()
            if DISPLAY:
                bullet.update_visual()
        if DISPLAY:
            self.fig.canvas.draw()
            plt.pause(0.01)

class Tank:
    def __init__(self, stage, x: float, y: float, starting_side_is_left=True, green=True):
        self.type = 'green' if green else 'brown'
        self.stage = stage
        self.stage.tanks.append(self)
        self.speed = 1
        self.x = x
        self.y = y
        if DISPLAY:
            self.color = 'green' if green else 'brown'
            self.visual = patches.Rectangle((self.x - 5, self.y - 5), TANK_SIZE, TANK_SIZE, color=self.color)
            self.stage.ax.add_patch(self.visual)
        self.rotation = 0 if starting_side_is_left else 180
        self.cannon = Cannon(self)
        self.agent = QLearningAgent([47, 47, 36], 3)  # State size: (x, y, angle), Action size: 3
        
    def tick(self):
        state = self.agent.get_state(self.x, self.y, calculate_angle_difference(
            calculate_angle(self.x, self.y, CENTER_X, CENTER_Y), self.rotation))
        action = self.agent.choose_action(state)
        self.perform_action(action)
        reward = self.calculate_reward()
        next_state = self.agent.get_state(self.x, self.y, calculate_angle_difference(
            calculate_angle(self.x, self.y, CENTER_X, CENTER_Y), self.rotation))
        self.agent.learn(state, action, reward, next_state)
    
    def perform_action(self, action):
        actions = [0, 0, 0, 0, 1, 0, 0]
        actions[action] = 1
        self.action(actions)
    
    def calculate_reward(self):
        distance_to_center = math.sqrt((self.x - CENTER_X) ** 2 + (self.y - CENTER_Y) ** 2)
        if 200 < self.x < 300 and 200 < self.y < 300:
            return 100  # Large reward for reaching the center
        return -distance_to_center  # Negative reward for distance
    
    def action(self, actions):
        if actions[0] == 1:
            self.rotateLeft()
        if actions[1] == 1:
            self.rotateRight()
        if actions[2] == 1:
            self.move(forward=True)
        if actions[3] == 1:
            self.move(forward=False)
        if actions[4] == 1:
            self.cannon.shoot()
        if actions[5] == 1:
            self.cannon.rotateLeft()
        if actions[6] == 1:
            self.cannon.rotateRight()
    
    def move(self, forward=True):
        if forward:
            self.x += self.speed * math.cos(math.radians(self.rotation))
            self.y -= self.speed * math.sin(math.radians(self.rotation))
        else:
            self.x -= self.speed * math.cos(math.radians(self.rotation))
            self.y += self.speed * math.sin(math.radians(self.rotation))
        self.x = max(min(self.x, MAX_X), MIN_X)
        self.y = max(min(self.y, MAX_Y), MIN_Y)
    
    def rotateRight(self):
        self.rotation = (self.rotation + 1.5) % 360
    
    def rotateLeft(self):
        self.rotation = (self.rotation - 1.5) % 360
    
    def update_visual(self):
        self.visual.set_xy((self.x - 5, self.y - 5))
        self.visual.angle = -self.rotation
    
    def destroy(self):
        if DISPLAY:
            self.visual.set_visible(False)
        self.stage.end_game()

class Cannon:
    max_shoot_cooldown = 100
    def __init__(self, parentTank: Tank):
        self.rotation = parentTank.rotation
        self.tank = parentTank
        self.cooldown = 0
    
    def rotateRight(self):
        self.rotation = (self.rotation + 1.5) % 360
    
    def rotateLeft(self):
        self.rotation = (self.rotation - 1.5) % 360
    
    def tick(self):
        if self.cooldown > 0:
            self.cooldown -= 1
    
    def shoot(self):
        if self.cooldown == 0:
            Bullet(self.tank, self.rotation)
            self.cooldown = self.max_shoot_cooldown

class Bullet:
    def __init__(self, parentTank: Tank, angle):
        self.tank = parentTank
        self.stage: Stage = parentTank.stage
        self.stage.bullets.append(self)
        self.x = parentTank.x
        self.y = parentTank.y
        self.speed = 2.5
        self.angle = angle
        if DISPLAY:
            self.visual = patches.Circle((self.x, self.y), 2.5, color='black')
            self.stage.ax.add_patch(self.visual)
    
    def move(self):
        self.x += self.speed * math.cos(math.radians(self.angle))
        self.y -= self.speed * math.sin(math.radians(self.angle))
        if self.is_out_of_bounds():
            self.destroy()
        else:
            self.check_collision()
            if DISPLAY:
                self.update_visual()
    
    def is_out_of_bounds(self):
        return self.x < MIN_X or self.x > MAX_X or self.y < MIN_Y or self.y > MAX_Y
    
    def check_collision(self):
        for tank in self.stage.tanks:
            if tank.type != self.tank.type:
                if abs(tank.x - self.x) < 5 and abs(tank.y - self.y) < 5:
                    print(f"{self}: collision with {tank}")
                    tank.destroy()
                    self.destroy()
                    return True
        return False
    
    def destroy(self):
        if DISPLAY:
            self.visual.set_visible(False)
        self.stage.bullets.remove(self)
    
    def update_visual(self):
        self.visual.set_center((self.x, self.y))

def main():
    stage = Stage()
    greenTank = Tank(stage, 50, random.randint(30, 470), True, True)
    brownTank = Tank(stage, 450, random.randint(30, 470), False, False)
    
    while stage.running:
        stage.tick()
    if DISPLAY:
        plt.show()

if __name__ == "__main__":
    main()
