from gym_derk.envs import DerkEnv

env = DerkEnv(home_team=[{'slots': ['Talons', None, None]}], away_team=[{'slots': ['Talons', None, None]}])

for t in range(1):
  observation_n = env.reset()
  print(observation_n)
  while True:
    action_n = [env.action_space.sample() for i in range(env.n_agents)]
    new_action_n = []
    for a in action_n:
        new_action = []
        new_action.append(a[0])
        new_action.append(a[1])
        new_action.append(a[2])
        new_action.append(1)
        new_action.append(a[4])
        new_action_n.append(new_action)
    observation_n, reward_n, done_n, info = env.step(new_action_n)
    print(observation_n)
    
    if all(done_n):
      print("Episode finished")
      break
env.close()