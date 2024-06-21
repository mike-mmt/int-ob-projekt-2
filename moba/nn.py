from gym_derk.envs import DerkEnv
from gym_derk import ObservationKeys, ActionKeys, TeamStatsKeys
import numpy as np
import gym
import math
import os.path
import os


weights_folder_path = 'weights/simple_roles-3-150-episodes'

weights_filepath =  weights_folder_path + '/weights.npy'
biases_filepath = weights_folder_path + '/biases.npy'

if not os.path.exists(weights_folder_path):
   os.makedirs(weights_folder_path)


reward_function_fighter = {
    "damageEnemyUnit": 4,
    "damageEnemyStatue": 1,
    "friendlyFire": -1,
    # "healEnemy": -10,
    "fallDamageTaken": -2,
    "healTeammate1": 1,
    "healTeammate2": 1,
    "healEnemy": -1,
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
healer_loadout = ['Talons','HealingGland', None]

env = DerkEnv(
    turbo_mode=False,
    mode='normal',
    n_arenas=1,
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

  def copy_and_mutate(self, network, mr=0.1):
    self.weights = np.add(network.weights, np.random.normal(size=self.weights.shape) * mr)
    self.biases = np.add(network.biases, np.random.normal(size=self.biases.shape) * mr)

weights = np.load(weights_filepath) if os.path.isfile(weights_filepath) else None
biases = np.load(biases_filepath) if os.path.isfile(biases_filepath) else None

networks = [Network(weights, biases) for i in range(env.n_agents)]

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
    
    # for o in observation_n:
      # if o[ObservationKeys.Stuck.value] > 0:
        # print('stuck')
    
    
    processed_reward_n = process_rewards(observation_n, old_observation_n, reward_n, action_n, old_action_n)
    
    old_action_n = action_n.copy()
    
    for r in range(len(reward_n)):
      r1 = reward_n[r]
      r2 = processed_reward_n[r]
      diff = r2 - r1
      env.total_reward[r] += diff
    
    
    if all(done_n):
        print("Episode finished")
        # print('observation[0]', observation_n[0])
        # print('reward_n', reward_n.shape)
        print('total reward', env.total_reward.shape, env.total_reward)
        break
  if env.mode == 'train':
    reward_n = env.total_reward
    # print(reward_n)
    top_network_i = np.argmax(reward_n)
    top_network = networks[top_network_i].clone()
    for network in networks:
      network.copy_and_mutate(top_network)
    print('top reward', reward_n[top_network_i])
    np.save(weights_filepath, top_network.weights)
    np.save(biases_filepath, top_network.biases)
env.close()
