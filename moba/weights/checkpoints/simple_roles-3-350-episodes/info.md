### episodes = ~350

```python
reward_function = {
    "damageEnemyUnit": 3,
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

loadout = ['Talons','HealingGland', None]

home_team=[
        {'slots': loadout, 'rewardFunction': reward_function, 'primaryColor': '#be2525'},
        {'slots': healer_loadout, 'rewardFunction': reward_function, 'primaryColor': '#f66b4e'},
        {'slots': healer_loadout, 'rewardFunction': reward_function, 'primaryColor': '#52d752'}
    ],
```
