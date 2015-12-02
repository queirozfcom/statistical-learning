
import numpy as np

def plotv(v):
	np.set_printoptions(formatter={'float':'{: 0.4f}'.format})
	v = v.reshape((5,5))
	print(v)