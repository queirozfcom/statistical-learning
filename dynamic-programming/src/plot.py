
import numpy as np

def plotv(v):
    float_formatter = lambda x: " %06.3f "%x
    np.set_printoptions(formatter={'float_kind':float_formatter})
    v = v.reshape((5,5))
    print(v)

def plotpi(pi):
    copy = np.copy(pi).ravel()
    moves = np.array(list(map(_index_to_move,copy)))
    moves = moves.reshape((5,5))
    print(moves)

def _index_to_move(index):
    if(index == 1):
        return("  up  ")
    elif index == 2:
        return("right ")
    elif index == 3:
        return(" down ")
    elif index == 4:
        return(" left ")
    else:
        raise ValueError("Unknown move: {0}".format(index))