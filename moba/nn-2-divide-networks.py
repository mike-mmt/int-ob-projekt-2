from typing import List
from gym_derk.envs import DerkEnv
from gym_derk import ObservationKeys, ActionKeys, TeamStatsKeys
import numpy as np
import gym
import math
import os.path
import os

USE_BEST = True

TRAIN = False
turbomode = True if TRAIN == True else False
narenas = 256 if TRAIN == True else 1
mode = 'train' if TRAIN == True else 'normal'

weights_folder_path = 'weights/simplest-with-statue/'


weights_filepath =  weights_folder_path + f'/{'best_' if USE_BEST else ''}'
biases_filepath = weights_folder_path + f'/{'best_' if USE_BEST else ''}'
weights_fln = 'weights.npy'
biases_fln = 'biases.npy'

if not os.path.exists(weights_folder_path):
   os.makedirs(weights_folder_path)


reward_function_fighter = {
    "damageEnemyUnit": 2,
    "damageEnemyStatue": 7,
    "friendlyFire": -1,
    # "healEnemy": -10,
    "fallDamageTaken": -100,
    # "healTeammate1": 1,
    # "healTeammate2": 1,
    # "healEnemy": -1,
    # "timeSpentAwayTerritory": -0.1,
    # "timeSpentAwayBase": -0.1,
    # "timeSpentHomeTerritory": -0.1,
    # "timeSpentHomeBase": -0.1,
    # "healFriendlyStatue": 4,
    # "timeScaling": 0.8
}

reward_function_healer = {
    "healTeammate1": 3,
    "healTeammate2": 3,
    # "healFriendlyStatue": 20,
    "healEnemy": -1,
    "damageEnemyUnit": 1,
    # "damageEnemyStatue": 15,
    # "killEnemyStatue": 15,
    # "killEnemyUnit": 15,
    "fallDamageTaken": -2,
    "timeSpentAwayTerritory": -0.1,
    # "timeSpentAwayBase": -0.1,
    "timeSpentHomeTerritory": 0.1,
    # "timeSpentHomeBase": -0.1,
}

fighter_loadout = ['Talons', None, None]
# healer_loadout = ['Talons','HealingGland', None]
healer_loadout = ['Talons', None, None]

env = DerkEnv(
    turbo_mode=turbomode,
    mode=mode,
    n_arenas=narenas,
    reward_function=reward_function_fighter,
    home_team=[
        {'slots': healer_loadout, 'rewardFunction': reward_function_fighter, 'primaryColor': '#be2525'},
        {'slots': healer_loadout, 'rewardFunction': reward_function_fighter, 'primaryColor': '#f66b4e'},
        {'slots': healer_loadout, 'rewardFunction': reward_function_fighter, 'primaryColor': '#52d752'}
    ],
    away_team=[
        {'slots': healer_loadout, 'rewardFunction': reward_function_fighter, 'primaryColor': '#c752d7'},
        {'slots': healer_loadout, 'rewardFunction': reward_function_fighter, 'primaryColor': '#d95bbe'},
        {'slots': healer_loadout, 'rewardFunction': reward_function_fighter, 'primaryColor': '#7deeff'}
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

  def copy_and_mutate(self, network, mr=0.01):
    self.weights = np.add(network.weights, np.random.normal(size=self.weights.shape) * mr)
    self.biases = np.add(network.biases, np.random.normal(size=self.biases.shape) * mr)

### load weights
networks = [Network() for i in range(env.n_agents)]

if not USE_BEST:
  w1 = np.load(weights_folder_path + 'n1' + weights_fln) if os.path.isfile(weights_folder_path + 'n1' + weights_fln) else None
  b1 = np.load(weights_folder_path + 'n1' + biases_fln) if os.path.isfile(weights_folder_path + 'n1' + biases_fln) else None
  w2 = np.load(weights_folder_path + 'n2' + weights_fln) if os.path.isfile(weights_folder_path + 'n2' + weights_fln) else None
  b2 = np.load(weights_folder_path + 'n2' + biases_fln) if os.path.isfile(weights_folder_path + 'n2' + biases_fln) else None
  w3 = np.load(weights_folder_path + 'n3' + weights_fln) if os.path.isfile(weights_folder_path + 'n3' + weights_fln) else None
  b3 = np.load(weights_folder_path + 'n3' + biases_fln) if os.path.isfile(weights_folder_path + 'n3' + biases_fln) else None
  w4 = np.load(weights_folder_path + 'n4' + weights_fln) if os.path.isfile(weights_folder_path + 'n4' + weights_fln) else None
  b4 = np.load(weights_folder_path + 'n4' + biases_fln) if os.path.isfile(weights_folder_path + 'n4' + biases_fln) else None
  w5 = np.load(weights_folder_path + 'n5' + weights_fln) if os.path.isfile(weights_folder_path + 'n5' + weights_fln) else None
  b5 = np.load(weights_folder_path + 'n5' + biases_fln) if os.path.isfile(weights_folder_path + 'n5' + biases_fln) else None
  w6 = np.load(weights_folder_path + 'n6' + weights_fln) if os.path.isfile(weights_folder_path + 'n6' + weights_fln) else None
  b6 = np.load(weights_folder_path + 'n6' + biases_fln) if os.path.isfile(weights_folder_path + 'n6' + biases_fln) else None
else:
  w1 = np.load(weights_folder_path + 'n1best_' + weights_fln) if os.path.isfile(weights_folder_path + 'n1best_' + weights_fln) else None
  b1 = np.load(weights_folder_path + 'n1best_' + biases_fln) if os.path.isfile(weights_folder_path + 'n1best_' + biases_fln) else None
  w2 = np.load(weights_folder_path + 'n2best_' + weights_fln) if os.path.isfile(weights_folder_path + 'n2best_' + weights_fln) else None
  b2 = np.load(weights_folder_path + 'n2best_' + biases_fln) if os.path.isfile(weights_folder_path + 'n2best_' + biases_fln) else None
  w3 = np.load(weights_folder_path + 'n3best_' + weights_fln) if os.path.isfile(weights_folder_path + 'n3best_' + weights_fln) else None
  b3 = np.load(weights_folder_path + 'n3best_' + biases_fln) if os.path.isfile(weights_folder_path + 'n3best_' + biases_fln) else None
  w4 = np.load(weights_folder_path + 'n4best_' + weights_fln) if os.path.isfile(weights_folder_path + 'n4best_' + weights_fln) else None
  b4 = np.load(weights_folder_path + 'n4best_' + biases_fln) if os.path.isfile(weights_folder_path + 'n4best_' + biases_fln) else None
  w5 = np.load(weights_folder_path + 'n5best_' + weights_fln) if os.path.isfile(weights_folder_path + 'n5best_' + weights_fln) else None
  b5 = np.load(weights_folder_path + 'n5best_' + biases_fln) if os.path.isfile(weights_folder_path + 'n5best_' + biases_fln) else None
  w6 = np.load(weights_folder_path + 'n6best_' + weights_fln) if os.path.isfile(weights_folder_path + 'n6best_' + weights_fln) else None
  b6 = np.load(weights_folder_path + 'n6best_' + biases_fln) if os.path.isfile(weights_folder_path + 'n6best_' + biases_fln) else None
  
if w1 is not None and b1 is not None and w2 is not None and b2 is not None and w3 is not None and b3 is not None and w4 is not None and b4 is not None and w5 is not None and b5 is not None and w6 is not None and b6 is not None:
  for i in range(0, len(networks)-6, 6):
    networks[i] = Network(w1, b1)
    networks[i+1] = Network(w2, b2)
    networks[i+2] = Network(w3, b3)
    networks[i+3] = Network(w4, b4)
    networks[i+4] = Network(w5, b5)
    networks[i+5] = Network(w6, b6)



###
best_reward = - 10000

for e in range(100000):
  # steps = 0
  observation_n = env.reset()
  print('Episode', e)
  
  old_action_n = None
  
  # for o in observation_n:
    # print(o)
  
  while True:
    action_n = [networks[i].forward(observation_n[i]) for i in range(env.n_agents)]
    
    old_observation_n = observation_n.copy()
    observation_n, reward_n, done_n, info = env.step(action_n)
    
    #
    old_action_n = action_n.copy()
    
   
    
    if all(done_n):
        print("Episode finished")
        # print('observation[0]', observation_n[0])
        # print('reward_n', reward_n.shape)
        print('total reward', env.total_reward.shape, env.total_reward)
        break
  if env.mode == 'train':
    

    best_team_reward_i = np.argmax(env.team_stats[:, TeamStatsKeys.Reward.value])
    best_team_reward = env.team_stats[best_team_reward_i, TeamStatsKeys.Reward.value]
    
    print("LEN", len(env.team_stats))
    print("LEN NETWORKS", len(networks))
    # best_team_reward = np.max(env.team_stats[:, TeamStatsKeys.Reward.value])
    print('best_team_reward', best_team_reward)
    second_best_team_reward_i = np.argsort(env.team_stats[:, TeamStatsKeys.Reward.value])[-2]
    second_best_team_reward = np.sort(env.team_stats[:, TeamStatsKeys.Reward.value])[-2]
    print('second_best_team_reward', second_best_team_reward)
    # env.team_stats[no_team, TeamStatsKeys.Reward.value]
    
    n1 = networks[best_team_reward_i*3]
    n2 = networks[best_team_reward_i*3 + 1]
    n3 = networks[best_team_reward_i*3 + 2]
    n4 = networks[second_best_team_reward_i*3]
    n5 = networks[second_best_team_reward_i*3 + 1]
    n6 = networks[second_best_team_reward_i*3 + 2]
    
    for i in range(0, len(networks)-6, 6):
      networks[i].copy_and_mutate(n1)
      networks[i+1].copy_and_mutate(n2)
      networks[i+2].copy_and_mutate(n3)
      networks[i+3].copy_and_mutate(n4)
      networks[i+4].copy_and_mutate(n5)
      networks[i+5].copy_and_mutate(n6)
    
    np.save(weights_folder_path + 'n1' + weights_fln, n1.weights)
    np.save(weights_folder_path + 'n1' + biases_fln, n1.biases)
    np.save(weights_folder_path + 'n2' + weights_fln, n2.weights)
    np.save(weights_folder_path + 'n2' + biases_fln, n2.biases)
    np.save(weights_folder_path + 'n3' + weights_fln, n3.weights)
    np.save(weights_folder_path + 'n3' + biases_fln, n3.biases)
    np.save(weights_folder_path + 'n4' + weights_fln, n4.weights)
    np.save(weights_folder_path + 'n4' + biases_fln, n4.biases)
    np.save(weights_folder_path + 'n5' + weights_fln, n5.weights)
    np.save(weights_folder_path + 'n5' + biases_fln, n5.biases)
    np.save(weights_folder_path + 'n6' + weights_fln, n6.weights)
    np.save(weights_folder_path + 'n6' + biases_fln, n6.biases)
    
    if best_team_reward > best_reward:
      best_reward = best_team_reward
      np.save(weights_folder_path + 'n1best_' + weights_fln, n1.weights)
      np.save(weights_folder_path + 'n1best_'+ biases_fln, n1.biases)
      np.save(weights_folder_path + 'n2best_'+ weights_fln, n2.weights)
      np.save(weights_folder_path + 'n2best_'+ biases_fln, n2.biases)
      np.save(weights_folder_path + 'n3best_'+ weights_fln, n3.weights)
      np.save(weights_folder_path + 'n3best_'+ biases_fln, n3.biases)
      np.save(weights_folder_path + 'n4best_'+ weights_fln, n4.weights)
      np.save(weights_folder_path + 'n4best_'+ biases_fln, n4.biases)
      np.save(weights_folder_path + 'n5best_'+ weights_fln, n5.weights)
      np.save(weights_folder_path + 'n5best_'+ biases_fln, n5.biases)
      np.save(weights_folder_path + 'n6best_'+ weights_fln, n6.weights)
      np.save(weights_folder_path + 'n6best_'+ biases_fln, n6.biases)
    
    # reward_n = env.total_reward
    # print(reward_n)
    # top_network_i = np.argmax(reward_n)
    # top_network = networks[top_network_i].clone()
    # for network in networks:
      # network.copy_and_mutate(top_network)
    # print('top reward', reward_n[top_network_i])
    # np.save(weights_filepath, top_network.weights)
    # np.save(biases_filepath, top_network.biases)
    
    # if np.max(reward_n) > best_reward:
      # best_reward = np.max(reward_n)
      # np.save(weights_folder_path + '/best_weights', top_network.weights)
      # np.save(weights_folder_path + '/best_biases', top_network.biases)

env.close()
