import math
import numpy as np

from . import settings


# important states
s_a = 1
s_b = 3
s_a_prime = (4*5) + 1 
s_b_prime = (2*5) + 3 

# rewards
r_a = 10
r_b = 5
r_off_limits = -1

def observe(s,a):
    """
    Observes application of action _a_ at state _s_.

    Returns the reward _r_ and the next state _s_prime_
    """

    x,y = _get_position_from_index(s)

    settings.num_observations += 1

    if s == s_a: # state a
        return(r_a,s_a_prime)
    elif s == s_b: # state b
        return(r_b,s_b_prime)
    else: # neither a nor b
        # obtain new coordinates    
        if a == 1: # move down
            y -= 1
        elif a == 2: # move right
            x += 1
        elif a == 3: # move up
            y += 1
        elif a == 4: # move left
            x -= 1
        else:
            raise ValueError("invalid action: '{0}' valid actions are [1,2,3,4]".format(a))       
        
        r = 0

        if x < 0:
            r = r_off_limits
            x = 0
        elif x > 4:
            r = r_off_limits
            x = 4
        elif y < 0:
            r = r_off_limits
            y = 0
        elif y > 4:
            r = r_off_limits
            y = 4   

    s_prime = _get_index_from_position(x,y)

    print(s_prime)

    return (r,s_prime)

def evaluate(pi,gamma,theta):
    """
    Evaluates policy _pi_ and returns its value function _V_
    """

    # initial values
    V = np.zeros(25)

    states = np.array([i for i in range(0,25)])

    for s in states:
        V[s] = _get_v(pi,s,theta,gamma)

    return(V)

def improve(v):
    """
    Improves
    """

def _get_v(pi,s,theta,gamma):
    """
    Recursively computes the value of a state _s_ under a certain policy _pi_.
    """

    def _iter_get_next(current_state,k):
        """
        Apply the action indicated by the policy and add the reward obtained,
        discounting by a factor of gamme to the power of k
        """

        multiplier = math.pow(gamma,k)

        if multiplier < theta:
            # recursion base case
            return(0)
        else:
            # recursive step - explore next application of the policy
            next_reward,next_state = observe(current_state,pi[current_state])

            new_reward = next_reward

            return(new_reward + (multiplier*_iter_get_next(next_state,k+1)) )

    v_final = _iter_get_next(s,1)

    return(v_final)  

def _get_position_from_index(index):
    """
    Returns the point (x,y) in the gridworld that represents index _index_.

    Note that the origin (0,0) of the gridworld is at the upper left.

    Example: _get_position_from_index(11) = (1,2)
    """
    
    x = index % 5
    y = math.floor(index / 5)

    return (x,y)

def _get_index_from_position(x,y):
    """
    Returns the index for the state that is located at (x,y) in the gridworld

    The opposite of function _get_position_from_index.

    Example: _get_index_from_position(1,2) = 11
    """

    index = (5 * y) + x

    return(index)
