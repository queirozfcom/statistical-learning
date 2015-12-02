import math
import numpy as np

from src import settings
from src.gridworld import evaluate,improve
from src.plot import plotv,plotpi

settings.init() # set num_observations global variable

gamma = 0.90
theta = math.pow(10,-5)

possible_actions = [1,2,3,4]

# initial value for policy: random sample of the possible actions
old_policy = np.random.choice(possible_actions,25)
policy = np.random.choice(possible_actions,25)

# initial state values: all zeros
V = np.zeros(25)

while True:
    old_policy = np.copy(policy)

    V = evaluate(policy,V,gamma,theta,possible_actions)

    policy = improve(V,policy,gamma,possible_actions)

    if np.array_equal(old_policy,policy):
        break

print("\n\n=============================================\n")
print("Total number of observations: {0}\n".format(settings.num_observations))
print("Final value function:\n")
plotv(V)
print("\nFinal policy: \n")
plotpi(policy)
print("\n\n=============================================\n\n")

