import numpy as np
import os
import random
from gym_derk.envs import DerkEnv
from gym_derk import ObservationKeys, ActionKeys

train = False
mode = 'train' if train == True else 'normal'
n_arenas = 4 if train == True else 1
turbo_mode = True if train == True else False

# Constants
NUM_EPISODES = 100000
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
EPSILON = 1.0 if train == True else 0.1
EPSILON_DECAY = 0.02
weights_folder_path = 'weights/q_learning'
TRANSFER_BEST = True

important_observation_indices = [
    ObservationKeys.Enemy1Distance.value,
    ObservationKeys.Enemy2Distance.value,
    ObservationKeys.Enemy3Distance.value,
    ObservationKeys.HeightFront1.value
]

# File paths
q_table_filepath = os.path.join(weights_folder_path, 'q_table4.npy')
q_table_load_filepath = os.path.join(weights_folder_path, 'q_table3.npy')

# Ensure weights folder exists
if not os.path.exists(weights_folder_path):
    os.makedirs(weights_folder_path)

# Define reward functions
reward_function_fighter = {
    "damageEnemyUnit": 6,
    "damageEnemyStatue": 1.5,
    "friendlyFire": -1,
    "fallDamageTaken": -2,
    "healTeammate1": 1,
    "healTeammate2": 1,
    "healEnemy": -1,
}

reward_function_healer = {
    "healTeammate1": 3,
    "healTeammate2": 3,
    "healEnemy": -1,
    "damageEnemyUnit": 1,
    "fallDamageTaken": -2,
    "timeSpentAwayTerritory": -0.1,
    "timeSpentHomeTerritory": 0.1,
}

# Initialize environment
env = DerkEnv(
    turbo_mode=turbo_mode,
    mode=mode,
    n_arenas=n_arenas,
    reward_function=reward_function_fighter,
    home_team=[
        {'slots': ['Talons', 'HealingGland', None], 'rewardFunction': reward_function_fighter, 'primaryColor': '#be2525'},
        {'slots': ['Talons', 'HealingGland', None], 'rewardFunction': reward_function_fighter, 'primaryColor': '#f66b4e'},
        {'slots': ['Talons', 'HealingGland', None], 'rewardFunction': reward_function_fighter, 'primaryColor': '#52d752'}
    ],
    away_team=[
        {'slots': ['Talons', 'HealingGland', None], 'rewardFunction': reward_function_fighter, 'primaryColor': '#c752d7'},
        {'slots': ['Talons', 'HealingGland', None], 'rewardFunction': reward_function_fighter, 'primaryColor': '#d95bbe'},
        {'slots': ['Talons', 'HealingGland', None], 'rewardFunction': reward_function_fighter, 'primaryColor': '#7deeff'}
    ],
)

print(env.n_agents)



all_actions = []
for move_x in range(-2, 3):
    for rotate in range(-2, 3):
        for chase_focus in range(0, 3):
            for cast_slot in range(4):
                for focus in range(8):
                    all_actions.append((move_x*0.5, rotate*0.5, chase_focus*0.5, cast_slot, focus))

all_states = []
for d1 in range(3):
    for d2 in range(3):
        for d3 in range(3):
            for s in range(2):
                all_states.append((d1, d2, d3, s))
                
# Initialize Q-tables
# state_size = len(ObservationKeys)
state_size = len(all_states)
action_size = len(all_actions)   # Number of actions (MoveX, Rotate, ChaseFocus, CastSlot, Focus)
print(f"State size: {state_size}, Action size: {action_size}")
q_tables = [np.zeros((state_size, action_size)) for _ in range(env.n_agents)]
print(np.shape(q_tables))
                        
def compress_distance_state(state):
    if state <= 0.15:
        return 2
    elif state > 0.15 and state < 0.75:
        return 1
    else:
        return 0

def compress_height_state(state):
    if state > 0.5:
        return 1
    else:
        return 0



def get_important_observation(observation):
    im = []
    for i in important_observation_indices:
        if i == ObservationKeys.HeightFront1.value:
            im.append(compress_height_state(observation[i]))
        else:
            im.append(compress_distance_state(observation[i]))
    return im

def find_matching_state_index(state):
    for state_i, s in enumerate(all_states):
        if np.array_equal(s, state):
            return state_i
    return None


# Load existing Q-tables if available
if os.path.isfile(q_table_filepath):
    q_tables = np.load(q_table_load_filepath, allow_pickle=True)

def get_action_i(q_table, state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return np.random.randint(action_size)
    matching_state_index = find_matching_state_index(state) 
    return np.argmax(q_table[matching_state_index])

def find_action_index(action):
    for i, a in enumerate(all_actions):
        if np.array_equal(a, action):
            return i
    return None

for e in range(NUM_EPISODES):
    observation_n = env.reset()
    total_reward = np.zeros(env.n_agents)
    print(f'Episode {e}')

    while True:
        actions = []
        old_observation_n = observation_n.copy()
        for i in range(env.n_agents):
            state = tuple(old_observation_n[i])
            
            state = get_important_observation(state)
            
            action_i = get_action_i(q_tables[i], state, EPSILON)
            action = all_actions[action_i]
            actions.append(action)

        # observation_n is a list of lists (observations) for each agent, shape: (n_agents, n_features)
        # reward_n is a list of rewards for each agent, shape: (n_agents,)
        observation_n, reward_n, done_n, _ = env.step(actions)
        total_reward += reward_n

        for i in range(env.n_agents):
            state = tuple(old_observation_n[i])
            state = get_important_observation(state)
            state_i = find_matching_state_index(state)
            action = actions[i]
            action_i = find_action_index(action)
            next_state = tuple(observation_n[i])
            next_state = get_important_observation(next_state)
            next_state_i = find_matching_state_index(next_state)
            reward = reward_n[i]
            
            if (state[0] == 2 or state[1] == 2 or state[3] == 2) and (action[3] == 1):
                reward += 6
            elif (state[0] == 2 or state[1] == 2 or state[3] == 2) and (action[3] == 0):
                reward -= 3
                
            if state[0] == 2 or state[1] == 2 or state[3] == 2:
                reward += 3
            elif state[0] == 1 or state[1] == 1 or state[3] == 1:
                reward += 1
            
            # done = done_n[i]

            # Q-learning update
            best_next_action = np.argmax(q_tables[i][next_state_i])
            td_target = reward + DISCOUNT_FACTOR * q_tables[i][next_state_i][best_next_action]
            td_error = td_target - q_tables[i][state_i][action_i]
            q_tables[i][state_i][action_i] += LEARNING_RATE * td_error

        if all(done_n):
            print(f"Episode finished with total reward: {total_reward}")
            break

    # Save Q-tables after each episode
    if TRANSFER_BEST == True:
        best_q_table_i = np.argmax(total_reward)
        best_q_table = q_tables[best_q_table_i]
        q_tables = [best_q_table for _ in range(env.n_agents)]
    
    np.save(q_table_filepath, q_tables)
    EPSILON = max(0.1, EPSILON - EPSILON_DECAY)
env.close()
