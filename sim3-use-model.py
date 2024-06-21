import random
import math
from typing import List
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from keras.models import load_model
import numpy as np

# model = load_model('tanks-dqn-t1.keras')
# model.summary()

qtable = np.load('qtable3.npy')
states = []
for x in range(16):
    for y in range(16):
        for angle in range(-10+1, 10):
            states.append([x, y, angle])
            
print("Size of the array: ",
      qtable.size)
 
print("Memory size of one array element in bytes: ",
      qtable.itemsize)
 
# memory size of numpy array in bytes
print("Memory size of numpy array in bytes:",
      qtable.size * qtable.itemsize)
# all_actions = []
# for i in range(2):
#     for j in range(2):
#         for k in range(2):
#             all_actions.append([i, j, k])

# myTank.x,myTank.y,myTank.rotation,myTank.cannonRotation,myTank.velocityX,myTank.velocityY,myTank.accelerationX,myTank.accelerationY,myTank.shootCooldown,
# myTank.controls.turnLeft,myTank.controls.turnRight,myTank.controls.goForward,myTank.controls.goBack,myTank.controls.shoot,myTank.controls.cannonLeft,myTank.controls.cannonRight,
# myBullet1.x,myBullet1.y,myBullet1.velocityX,myBullet1.velocityY,myBullet2.x,myBullet2.y,myBullet2.velocityX,myBullet2.velocityY,myBullet3.x,myBullet3.y,myBullet3.velocityX,myBullet3.velocityY,
# enemyTank.x,enemyTank.y,enemyTank.rotation,enemyTank.cannonRotation,enemyTank.velocityX,enemyTank.velocityY,enemyTank.accelerationX,enemyTank.accelerationY,enemyTank.shootCooldown,
# enemyBullet1.x,enemyBullet1.y,enemyBullet1.velocityX,enemyBullet1.velocityY,enemyBullet2.x,enemyBullet2.y,enemyBullet2.velocityX,enemyBullet2.velocityY,enemyBullet3.x,enemyBullet3.y,enemyBullet3.velocityX,enemyBullet3.velocityY,
# currentGameTime
MIN_X = 15.0
MAX_X = 485.0
MIN_Y = 15.0
MAX_Y = 485.0

NUM_X = 16
NUM_Y = 16
NUM_ANGLE_DIFF = 10

original_range = 485 - 15 + 1
scaling_factor = (NUM_X - 1) / original_range

angle_scaling_factor = (NUM_ANGLE_DIFF - 1) / 180

DISPLAY = True

def create_position_actions():
    all_actions = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                all_actions.append([i, j, k])
    return all_actions

def create_aiming_actions():
    all_actions = []
    for i in range(2):
                for j in range(2):
                    all_actions.append([i, j])
    return all_actions

# def prepare_pos_state(state):
#         angle = calculate_angle(state[0], state[1], (NUM_X-1)/2, (NUM_Y-1)/2)
#         # cannonAngleDifference = (360 + angle - state.myTank.cannon.rotation) % 360
#         tankAngleDifference = ((angle - state[2]) + 180) % 360 - 180
        
#         scaled_x = int(round((state[0] - 15) * scaling_factor))
#         scaled_y = int(round((state[1] - 15) * scaling_factor))
        
#         return [scaled_x, scaled_y, tankAngleDifference]
    
def prepare_pos_state(state):
    angle = calculate_angle(state[0], state[1], (NUM_X-1)/2, (NUM_Y-1)/2)
    # cannonAngleDifference = (360 + angle - self.myTank.cannon.rotation) % 360
    tankAngleDifference = ((angle - state[2]) + 180) % 360 - 180
    
    
    scaled_x = int(round((state[0] - 15) * scaling_factor))
    scaled_y = int(round((state[1] - 15) * scaling_factor))
    
    scaled_angle_diff = int(round((tankAngleDifference) * angle_scaling_factor))
    
    # print(f"angleDiff: {tankAngleDifference}, scaled: {scaled_angle_diff}")
    
    return [scaled_x, scaled_y, scaled_angle_diff]

def preprocess_pos_state(state):
    x_norm = state[0] / (NUM_X-1)
    y_norm = state[1] / (NUM_Y-1)
    angle_norm = state[2] / (NUM_ANGLE_DIFF)
    return [x_norm, y_norm, angle_norm]

# def prepare_aim_state(self):
#     angle = calculate_angle(self.myTank.x, self.myTank.y, self.enemyTank.x, self.enemyTank.y)
#     # cannonAngleDifference = (360 + angle - self.myTank.cannon.rotation) % 360
#     cannonAngleDifference = ((angle - self.myTank.cannon.rotation) + 180) % 360 - 180
    
#     scaled_x = int(round((self.myTank.x - 15) * scaling_factor))
#     scaled_y = int(round((self.myTank.y - 15) * scaling_factor))
#     scaled_enemy_x = int(round((self.enemyTank.x - 15) * scaling_factor))
#     scaled_enemy_y = int(round((self.enemyTank.y - 15) * scaling_factor))
#     return [scaled_x, scaled_y, cannonAngleDifference, scaled_enemy_x, scaled_enemy_y]

# def preprocess_aim_state(self, state):
#     x_norm = state[0] / (NUM_X-1)
#     y_norm = state[1] / (NUM_Y-1)
#     angle_norm = state[2] / (NUM_ANGLE_DIFF)
#     enemy_x_norm = state[3] / (NUM_X-1)
#     enemy_y_norm = state[4] / (NUM_Y-1)
#     return [x_norm, y_norm, angle_norm, enemy_x_norm, enemy_y_norm]

def random_01():
    return random.randint(0, 1)

def calculate_angle(x1, y1, x2, y2):
    return (360 + math.atan2(y2 - y1, x2 - x1) * 180 / math.pi) % 360

possible_actions = create_position_actions()
print(possible_actions)


class Stage():
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
            tank = self.tanks[i]
            if i == 0:
                state = [tank.x, tank.y, tank.rotation]
                pos_state = prepare_pos_state(state)
                
                ### model
                # pos_state = preprocess_pos_state(pos_state)
                # pos_state = np.array(pos_state).reshape(1, 3)
                # final_state = np.reshape(pos_state, (1, 3))
                # prediction = model.predict(pos_state, verbose=0)
                # action_index = np.argmax(prediction)
                ###
                state_index = states.index(pos_state)
                action_index = np.argmax(qtable[state_index])
                                
                action = possible_actions[action_index] + [0, 1, 0, 0]
                tank.action(action)
                print(f"Tank {i} action: {action}")
            else:
                tank.action([0, 0, 0, 0, 1, 0, 0])
            tank.cannon.tick()
            if DISPLAY: tank.update_visual()
        for bullet in self.bullets:
            bullet.move()
            if DISPLAY: bullet.update_visual()
        # print(f"tick: T1[{self.tanks[0].x}, {self.tanks[0].y}], T2[{self.tanks[1].x}, {self.tanks[1].y}], ")
        if DISPLAY:
            self.fig.canvas.draw()
            plt.pause(0.01)

class Tank():
    def __init__(self, stage, x: float, y: float, starting_side_is_left=True, green=True):
        super(Tank, self).__init__()
        self.type = 'green' if green else 'brown'
        self.stage = stage
        self.stage.tanks.append(self)
        self.speed = 1
        self.x = x
        self.y = y
        if DISPLAY:
            self.color = 'green' if green else 'brown'
            self.visual = patches.Rectangle((self.x-5, self.y-5), 10, 10, color=self.color)
            self.stage.ax.add_patch(self.visual)
        if starting_side_is_left:
            self.rotation = 0
        else:
            self.rotation = 180
        self.cannon = Cannon(self)
            
            
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
        if self.x < MIN_X:
            self.x = MIN_X
        elif self.x > MAX_X:
            self.x = MAX_X
        if self.y < MIN_Y:
            self.y = MIN_Y
        elif self.y > MAX_Y:
            self.y = MAX_Y

    def rotateRight(self):
        self.rotation = (self.rotation + 1.5) % 360
        
    def rotateLeft(self):
        self.rotation = (self.rotation - 1.5) % 360

    def update_visual(self):
        self.visual.set_xy((self.x-5, self.y-5))
        self.visual.angle = -self.rotation
            
    def destroy(self):
        if DISPLAY: self.visual.set_visible(False)
        self.stage.end_game()
            
        
class Cannon():
    max_shoot_cooldown = 100
    def __init__(self, parentTank: Tank):
        super(Cannon, self).__init__()
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
            bullet = Bullet(self.tank, self.rotation)
            self.cooldown = self.max_shoot_cooldown

class Bullet():
    def __init__(self, parentTank: Tank, angle):
        super(Bullet, self).__init__()
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
            if DISPLAY: self.update_visual()
        
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
        if DISPLAY: self.visual.set_visible(False)
        self.stage.bullets.remove(self)
        
    def update_visual(self):
        self.visual.set_center((self.x, self.y))

def random_point():
    x = random.randint(MIN_X, MAX_X)
    y = random.randint(MIN_Y, MAX_Y)
    return x, y

def random_center():
    x = random.randint(233, 253)
    y = random.randint(233, 253)
    return x, y

def main():
    stage = Stage()
    greenTank = Tank(stage, 50, random.randint(30, 470), True, True)
    brownTank = Tank(stage, 450, random.randint(30, 470), False, False)
    
    # for i in range(10000):
    while stage.running:
        stage.tick()
        if not stage.running:
            break
    if DISPLAY: plt.show()

if __name__ == "__main__":
    main()