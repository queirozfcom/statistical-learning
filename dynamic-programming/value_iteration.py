import math
import numpy as np

from src import settings
from src.gridworld import evaluate,improve,update_value_greedy,get_policy_from_state_value_function
from src.plot import plotv,plotpi

settings.init() # set num_observations global variable

gamma = 0.90
theta = math.pow(10,-5)

possible_actions = [1,2,3,4]

states = np.array([i for i in range(0,25)])

# initial state values: all zeros
V = np.zeros(25)

while True:
    old_V = np.copy(V)
    for s in states:
        v = V[s]
        V[s] = update_value_greedy(s,possible_actions,old_V,gamma)

    # return true if every element in V and old_V are closer than THETA
    # to each other.    
    if np.allclose(old_V,V,rtol=0,atol=theta):
        break

policy = get_policy_from_state_value_function(V,possible_actions,gamma)

print("\n\n=============================================\n")
print("Total number of observations: {0}\n".format(settings.num_observations))
print("Final value function:\n")
plotv(V)
print("\nFinal policy: \n")
plotpi(policy)
print("\n\n=============================================\n\n")