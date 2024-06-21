from gym_derk.envs import DerkEnv
from gym_derk import ObservationKeys
import numpy as np
import gym
import math
import os.path
import os


train = True

folder_path = 'weights/simple-roles-1'

turbo_mode = True if train else False
mode = 'train' if train else 'normal'
n_arenas = 128 if train else 1

weights_filepath =  folder_path + '/weights.npy'
biases_filepath = folder_path + '/biases.npy'

if not os.path.exists(folder_path):
   os.makedirs(folder_path)


reward_function_fighter = {
    'damageEnemyUnit': 2,
    'friendlyFire': -1,
    'fallDamageTaken': -2,
    'timeSpentHomeBase': -0.1
}

reward_function_healer = {
    'damageEnemyUnit': 1,
    'healTeammate1': 2,
    'healTeammate2': 2,
    'healEnemy': -1,
    'timeSpentHomeTerritory': 0.1,
    'timeSpentEnemyTerritory': -0.1
}

fighter_loadout = ['Talons', None, None]
healer_loadout = ['Talons','HealingGland', None]

env = DerkEnv(
    turbo_mode=True,
    mode='train',
    n_arenas=256,
    reward_function=reward_function_fighter,
    home_team=[
        {'slots': fighter_loadout, 'rewardFunction': reward_function_fighter, 'primaryColor': '#be2525'},
        {'slots': fighter_loadout, 'rewardFunction': reward_function_fighter, 'primaryColor': '#f66b4e'},
        {'slots': healer_loadout, 'rewardFunction': reward_function_healer, 'primaryColor': '#52d752'}
    ],
    away_team=[
        {'slots': fighter_loadout, 'rewardFunction': reward_function_fighter, 'primaryColor': '#c752d7'},
        {'slots': fighter_loadout, 'rewardFunction': reward_function_fighter, 'primaryColor': '#d95bbe'},
        {'slots': healer_loadout, 'rewardFunction': reward_function_healer, 'primaryColor': '#7deeff'}
    ]
    )

print(env.n_agents)

def process_rewards(observation_n, rewards_n):
    new_rewards = []
    for i in range(len(rewards_n)):
        reward = rewards_n[i]
        reward -= 0.5 # time step penalty
        if observation_n[i][30] > 0:
            reward -= 5 # stuck
        
        new_rewards.append(reward)
    return np.array([r - 0.5 for r in rewards_n])


class Network:
  def __init__(self, weights=None, biases=None):
    self.network_outputs = 13
    if weights is None:
      weights_shape = (self.network_outputs, len(ObservationKeys))
      self.weights = np.random.normal(size=weights_shape)
    else:
      self.weights = weights
    if biases is None:
      self.biases = np.random.normal(size=(self.network_outputs))
    else:
      self.biases = biases

  def clone(self):
    return Network(np.copy(self.weights), np.copy(self.biases))

  def forward(self, observations):
    outputs = np.add(np.matmul(self.weights, observations), self.biases)
    casts = outputs[3:6]
    cast_i = np.argmax(casts)
    focuses = outputs[6:13]
    focus_i = np.argmax(focuses)
    return (
      math.tanh(outputs[0]), # MoveX
      math.tanh(outputs[1]), # Rotate
      max(min(outputs[2], 1), 0), # ChaseFocus
      (cast_i + 1) if casts[cast_i] > 0 else 0, # CastSlot
      (focus_i + 1) if focuses[focus_i] > 0 else 0, # Focus
    )

  def copy_and_mutate(self, network, mr=0.1):
    self.weights = np.add(network.weights, np.random.normal(size=self.weights.shape) * mr)
    self.biases = np.add(network.biases, np.random.normal(size=self.biases.shape) * mr)

weights = np.load(weights_filepath) if os.path.isfile(weights_filepath) else None
biases = np.load(biases_filepath) if os.path.isfile(biases_filepath) else None

networks = [Network(weights, biases) for i in range(env.n_agents)]

for e in range(100000):
  # steps = 0
  observation_n = env.reset()
  print("Observation", observation_n)
  print('Episode', e)
  while True:
    action_n = [networks[i].forward(observation_n[i]) for i in range(env.n_agents)]
    observation_n, reward_n, done_n, info = env.step(action_n)
    # reward_n = process_rewards(observation_n, reward_n)
    # steps += 1
    if all(done_n):
        print("Episode finished")
        # print('reward_n', reward_n.shape)
        break
  if env.mode == 'train':
    reward_n = env.total_reward
    print(reward_n)
    top_network_i = np.argmax(reward_n)
    top_network = networks[top_network_i].clone()
    for network in networks:
      network.copy_and_mutate(top_network)
    print('top reward', reward_n[top_network_i])
    np.save(weights_filepath, top_network.weights)
    np.save(biases_filepath, top_network.biases)
env.close()
