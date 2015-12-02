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
        if a == 1: # move up
            y -= 1
        elif a == 2: # move right
            x += 1
        elif a == 3: # move down
            y += 1
        elif a == 4: # move left
            x -= 1
        else:
            raise ValueError("invalid action: '{0}' for state {1} valid actions are [1,2,3,4]".format(a,s))       
        
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

    return (r,s_prime)

def evaluate(pi,V,gamma,theta,possible_actions):
    """
    Approximates the value function for a given policy.
    """


    new_V = np.copy(V)

    delta = np.float32(0.0)
    states = np.array([i for i in range(0,25)])

    while True:

        for s in states:
            v = new_V[s]

            reward,next_state = observe(s,pi[s])

            new_V[s] = reward + gamma * new_V[next_state]   
            
            delta = max(delta,np.abs(v-new_V[s]))

        if  delta > theta:
            break  

    return(new_V)        

def improve(V,pi,gamma,possible_actions):
    """
    Greedily improves the given deterministic policy _pi_. Computes _pi_prime_ which
    amounts to taking _pi_ and, for each state s and possible action in _possible_actions_,
    choosing the action that yields the best immediate reward + value for state _s_ (recursvely).

    Returns:
        The improved policy pi_prime
    """

    states = [i for i in range(0,25)]

    pi_prime = np.copy(pi)

    for s in states:
    
        rewards = []
        next_states = []

        for a in possible_actions:
            reward,next_state = observe(s,a)
            rewards.append(reward)
            next_states.append(next_state)
      
        next_state_values = [V[next_state] for next_state in next_states]

        pi_prime[s] = _get_best_action(gamma,rewards,next_state_values,possible_actions)

    return(pi_prime)

def update_value_greedy(s,possible_actions,old_V,gamma):
    """
    Updates the value for state _s_, according to the reward obtained 
    over all possible actions and the previous estimate of the value for 
    that state.   
    """

    rewards = np.array([])
    previous_estimates = np.array([])

    for a in possible_actions:
        reward,next_state = observe(s,a)
        rewards = np.append(rewards,reward)
        previous_estimate = old_V[next_state]
        previous_estimates = np.append(previous_estimates,previous_estimate)

    total_values = rewards + gamma*previous_estimates
    
    best_value = float("-inf")
    best_action = None

    for i,value in enumerate(total_values):
        if value > best_value:
            best_value  = value
            best_action = possible_actions[i]

    return(best_value)         

def get_policy_from_state_value_function(V,possible_actions,gamma):

    states = np.array([i for i in range(0,25)])

    # starting policy: empty
    policy = np.empty(25)

    # for every state _s_ just select the action that maximizes
    # reward + gamma*value(next_state)
    for s in states:
        rewards = np.array([])
        next_values = np.array([])

        for a in possible_actions:

            reward,next_state = observe(s,a)

            rewards = np.append(rewards,reward)

            next_value = gamma * V[next_state]
            next_values = np.append(next_values,next_value)

        best_action = None
        best_value = float("-inf")

        for i,value in enumerate(next_values):
            if value > best_value:
                best_value = value
                best_action = possible_actions[i]

        policy[s] = best_action       
            
    return(policy)        

def _get_v(pi,s,theta,gamma):
    """
    Recursively computes the value of a state _s_ under a certain policy _pi_.

    Future rewards are also taken into account, until the multiplier (_gamma_^k)
    falls below the threshold given by _theta_

    Returns: the value of state _s_ under policy _pi_.
    """

    def _iter_get_next(current_state,k):
        """
        Applies the action indicated by the policy and adds the reward 
        obtained, discounting by a factor of gamma to the power of k.
        """

        multiplier = math.pow(gamma,k)

        if multiplier < theta:
            # recursion base case
            return(0)
        else:
            # recursive step - explore next application of the policy
            reward,next_state = observe(current_state,pi[current_state])

            return(reward + (multiplier*_iter_get_next(next_state,k+1)) )

    v_final = _iter_get_next(s,0)

    return(v_final)  

def _get_position_from_index(index):
    """
    Returns the point (x,y) in the gridworld that represents index _index_.

    Note that the origin (0,0) of the gridworld is at the upper left, and that
    state indices begin at zero, i.e. position (0,0) refers to state zero.

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

def _get_best_action(gamma,rewards,next_state_values,possible_actions):
    """
    Returns the best action among _possible_actions_. The first action in _possible_actions_
    yields the first rewards in _rewards_ and the first state value in _state_values_.

    Returns:
        A single integer which represents the best action to follow given the inputs
    """

    total_values = np.array(rewards)+ (gamma*np.array(next_state_values))

    current_best_total_value = float("-inf")
    current_best_action      = None

    for i,total_value in enumerate(total_values):
        if total_value > current_best_total_value:
            current_best_total_value = total_value
            # actions start at 1
            current_best_action = possible_actions[i]

    return(current_best_action)        
            
