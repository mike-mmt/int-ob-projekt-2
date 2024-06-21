from typing import List
from gym_derk.envs import DerkEnv
from gym_derk import ObservationKeys, ActionKeys, TeamStatsKeys
import numpy as np
import gym
import math
import os.path
import os


weights_folder_path = 'weights/attackers-defenders/simple-2'

atk_folder_path = weights_folder_path + '/attackers'
def_folder_path = weights_folder_path + '/defenders'

attacker_weights_filepath =  atk_folder_path + '/weights.npy'
attacker_biases_filepath = atk_folder_path + '/biases.npy'

defender_weights_filepath =  def_folder_path + '/weights.npy'
defender_biases_filepath = def_folder_path + '/biases.npy'

if not os.path.exists(weights_folder_path):
  os.makedirs(weights_folder_path)
if not os.path.exists(atk_folder_path):
  os.makedirs(atk_folder_path)
if not os.path.exists(def_folder_path):
  os.makedirs(def_folder_path)


reward_function_attacker = {
    # "damageEnemyUnit": 1,
    "damageEnemyStatue": 1,
    # "killEnemyUnit": 0,
    "killEnemyStatue": 100,
    # "healFriendlyStatue": 4,
    # "healTeammate1": 1,
    # "healTeammate2": 1,
    # "timeSpentHomeBase": -100,
    # "timeSpentHomeTerritory": -25,
    # "timeSpentAwayTerritory": 25,
    # "timeSpentAwayBase": 150,
    # "damageTaken": 0,
    # "friendlyFire": -1,
    # "healEnemy": -1,
    "fallDamageTaken": -100,
    # "statueDamageTaken": 0,
    # "teamSpirit": 0,
    # "timeScaling": 0.8
}

reward_function_defender = {
    "damageEnemyUnit": 5,
    # "damageEnemyStatue": 0,
    "killEnemyUnit": 300,
    # "killEnemyStatue": 100,
    "healFriendlyStatue": 2,
    # "healTeammate1": 1,
    # "healTeammate2": 1,
    # "timeSpentHomeBase": -100, # x 150 = 150
    # "timeSpentHomeTerritory": -25, # x 150 = 0
    # "timeSpentAwayTerritory": 25, # x 150 = -75
    # "timeSpentAwayBase": 150, # x 150 = -150
    # "damageTaken": 0,
    "friendlyFire": -5,
    "healEnemy": -1,
    "fallDamageTaken": -200,
    # "statueDamageTaken": -3,
    # "teamSpirit": 0,
    # "timeScaling": 0.8
}

loadout = ['Talons','HealingGland', None]

env = DerkEnv(
    turbo_mode=False,
    mode='normal',
    n_arenas=1,
    reward_function={},
    home_team=[
        {'slots': loadout, 'rewardFunction': reward_function_attacker, 'primaryColor': '#be2525'},
        {'slots': loadout, 'rewardFunction': reward_function_attacker, 'primaryColor': '#f66b4e'},
        {'slots': loadout, 'rewardFunction': reward_function_attacker, 'primaryColor': '#52d752'}
    ],
    away_team=[
        {'slots': loadout, 'rewardFunction': reward_function_defender, 'primaryColor': '#c752d7'},
        {'slots': loadout, 'rewardFunction': reward_function_defender, 'primaryColor': '#d95bbe'},
        {'slots': loadout, 'rewardFunction': reward_function_defender, 'primaryColor': '#7deeff'}
    ],
    )

print(env.n_agents)

def process_rewards(observation_n, old_observation_n, rewards_n, action_n, old_action_n):
    new_rewards = []
    for i in range(len(rewards_n)):
        reward = rewards_n[i]
        # reward -= 0.3 # time step penalty
        if observation_n[i][ObservationKeys.Stuck.value] > 0:
          reward -= 3 # stuck
        if observation_n[i][ObservationKeys.Ability0Ready.value] > 0 and \
          old_observation_n[i][ObservationKeys.Ability0Ready.value] == 0 and \
          action_n[i][ActionKeys.CastingSlot.value] == 1:
            reward += 0.1 # ability used
        if observation_n[i][ObservationKeys.Ability0Ready.value] > 0 and \
          old_observation_n[i][ObservationKeys.Ability0Ready.value] > 0 and \
          action_n[i][ActionKeys.CastingSlot.value] == 0:
            reward -= 0.1 # ability not used
        # if old_action_n is not None:
        #     if old_action_n[i][ActionKeys.ChangeFocus.value] > 0 and action_n[i][ActionKeys.ChangeFocus.value] == 0:
        #         reward += 0.1
        #     elif old_action_n[i][ActionKeys.ChangeFocus.value] == 0 and action_n[i][ActionKeys.ChangeFocus.value] > 0:
        #         reward -= 0.1 # discourage changing focus
        # reward += action_n[i][ActionKeys.ChaseFocus.value] / 10
        
        new_rewards.append(reward)
    return np.array(rewards_n)


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

  def copy_and_mutate(self, network, mr=0.001):
    self.weights = np.add(network.weights, np.random.normal(size=self.weights.shape) * mr)
    self.biases = np.add(network.biases, np.random.normal(size=self.biases.shape) * mr)


attacker_weights = np.load(attacker_weights_filepath) if os.path.isfile(attacker_weights_filepath) else None
attacker_biases = np.load(attacker_biases_filepath) if os.path.isfile(attacker_biases_filepath) else None

networks = [Network(attacker_weights, attacker_biases) for i in range(env.n_agents)]

defender_weights = np.load(defender_weights_filepath) if os.path.isfile(defender_weights_filepath) else None
defender_biases = np.load(defender_biases_filepath) if os.path.isfile(defender_biases_filepath) else None

for i in range(3, env.n_agents-3, 3):
  # print(i, i+1, i+2)
  networks[i] = Network(defender_weights, defender_biases)
  networks[i+1] = Network(defender_weights, defender_biases)
  networks[i+2] = Network(defender_weights, defender_biases)
# networks = [Network(weights, biases) for i in range(env.n_agents)]


last_atk_reward = -1000
last_def_reward = -1000

for e in range(100000):
  # steps = 0
  observation_n = env.reset()
  print('Episode', e)
  
  
  
  # old_action_n = None
  
  while True:
    action_n = [networks[i].forward(observation_n[i]) for i in range(env.n_agents)]
    
    old_observation_n = observation_n.copy()
    observation_n, reward_n, done_n, info = env.step(action_n)
    
    # processed_reward_n = process_rewards(observation_n, old_observation_n, reward_n, action_n, old_action_n)
    
    # old_action_n = action_n.copy()
    
    # for r in range(len(reward_n)):
    #   r1 = reward_n[r]
    #   r2 = processed_reward_n[r]
    #   diff = r2 - r1
    #   env.total_reward[r] += diff
        
    if all(done_n):
        print("Episode finished")
        # print('observation[0]', observation_n[0])
        # print('reward_n', reward_n.shape)
        print('total reward', env.total_reward.shape, env.total_reward)
        break
  if env.mode == 'train':
    reward_n = env.total_reward
    
    
    attacker_reward_n = np.array([])
    defender_reward_n = np.array([])
    attacker_networks = np.array([])
    defender_networks = np.array([])
    
    # networks = [atk, atk, atk, def, def, def, atk, atk, atk, def, def, def, ...]
    for i in range(0, env.n_agents-6, 6):
      
      ### attacker
      attacker_reward_n = np.append(attacker_reward_n, reward_n[i:i+3])
      attacker_networks = np.append(attacker_networks, networks[i:i+3])
      
      top_reward_attacker_reward = np.max(attacker_reward_n)
      
      if top_reward_attacker_reward > last_atk_reward:
        last_atk_reward = top_reward_attacker_reward
        top_attacker_network_i = np.argmax(attacker_reward_n)
        top_attacker_network = attacker_networks[top_attacker_network_i].clone()
        for n in networks[i:i+3]:
          n.copy_and_mutate(top_attacker_network)
        np.save(attacker_weights_filepath, top_attacker_network.weights)
        np.save(attacker_biases_filepath, top_attacker_network.biases)
      ###
      
      ### defender
      defender_reward_n = np.append(defender_reward_n, reward_n[i+3:i+6])
      defender_networks = np.append(defender_networks, networks[i+3:i+6])
      
      top_defender_reward = np.max(defender_reward_n)
      
      if top_defender_reward > last_def_reward:
        last_def_reward = top_defender_reward
        top_defender_network_i = np.argmax(defender_reward_n)
        top_defender_network = defender_networks[top_defender_network_i].clone()
        for n in networks[i+3:i+6]:
          n.copy_and_mutate(top_defender_network)
        np.save(defender_weights_filepath, top_defender_network.weights)
        np.save(defender_biases_filepath, top_defender_network.biases)
            
    # for i in range(3, env.n_agents-3, 6):
      
    
    
    # top_defender_network_i = np.argmax(defender_reward_n)
    # top_defender_network = networks[top_defender_network_i].clone()
    
    # for network in networks[::2]:
    #   network.copy_and_mutate(top_attacker_network)
    # for network in networks[1::2]:
    #   network.copy_and_mutate(top_defender_network)
      
    print('last/top atk reward:', round(last_atk_reward,2), round(reward_n[top_attacker_network_i],2), 'last/top defender reward:', round(last_def_reward,2), round(reward_n[top_defender_network_i],2))
      
    
    
    
env.close()
