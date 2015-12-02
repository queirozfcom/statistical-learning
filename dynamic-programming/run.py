import math
import numpy as np

from src import settings
from src.gridworld import evaluate,improve
from src.plot import plotv

settings.init() # set num_observations global variable

gamma = 0.9 
theta = math.pow(10,-5)

possible_actions = [1,2,3,4]

# initial value for policy: random sample of the possible actions
last_policy = policy = np.random.choice(possible_actions,25)

# initial values: all zeros
v = np.zeros(25,dtype=np.int8)

# while np.not_equal(lastpi,pi).any():
#     lastpi = np.copy(pi)
#     v = evaluate(pi)
#     pi = improve(v)
last_policy = np.copy(policy)
v = evaluate(policy,gamma,theta)

print("\n\n=============================================\n")
print("Total number of observations: {0}\n".format(settings.num_observations))
print("value for each state given by the optimal value function:\n")
plotv(v)
print("\n\n=============================================\n\n")

