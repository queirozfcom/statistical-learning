
from src.gridworld import observe,evaluate,improve

gamma = 0.9 
theta = math.pow(10,-5)
observations = 0

lastpi = np.zeros(25)
pi = lastpi + 1

v = lastpi

while any(lastpi != pi):
    lastpi = copy(pi)
    v = evaluate(pi)
    pi = improve(v)
