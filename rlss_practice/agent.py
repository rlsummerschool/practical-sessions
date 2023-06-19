import numpy as np
from collections import defaultdict
from copy import deepcopy



class Agent:
    """Agent. Default policy is purely random."""
    def __init__(self, environment, policy=None, n_steps=100):
        if policy is not None:
            self.policy = policy
        else:
            self.policy = self.random_policy
        self.environment = environment
        self.n_steps = 100
            
    def random_policy(self, state):
        actions = self.environment.get_actions(state)
        if len(actions):
            probs = np.ones(len(actions)) / len(actions)
        else:
            probs = [1]
            actions = [None]
        return probs, actions

    def get_action(self, state):
        action = None
        probs, actions = self.policy(state)
        if len(actions):
            i = np.random.choice(len(actions), p=probs)
            action = actions[i]
        return action
    
    def get_episode(self, n_steps=None):
        """Get the states and rewards for an episode."""
        self.environment.reinit_state()
        if n_steps is None:
            n_steps = self.n_steps
        state = deepcopy(self.environment.state)
        states = [state] # add the initial state
        rewards = [0]
        for t in range(n_steps):
            action = self.get_action(state)
            reward, stop = self.environment.step(action)
            state = deepcopy(self.environment.state)
            states.append(state)
            rewards.append(reward)
            if stop:
                break
        return stop, states, rewards

    
class Player(Agent):
    """Player, specific agent for games."""
    def __init__(self, environment, policy=None, n_steps=100):
        super(Player, self).__init__(environment, policy, n_steps)   
                
    def random_policy(self, state):
        actions = self.environment.get_actions(state, player=1)
        if len(actions):
            probs = np.ones(len(actions)) / len(actions)
        else:
            probs = [1]
            actions = [None]
        return probs, actions
    

class OnlinePolicyEvaluation(Player):
    """Online policy evaluation. The agent interacts with the environment and learns the value function of its policy."""
    
    def __init__(self, environment, policy, gamma=1, alpha=0.1, n_steps=100):
        super(OnlinePolicyEvaluation, self).__init__(environment, policy, n_steps)   
        self.gamma = gamma # discount rate
        self.alpha = alpha # learning rate
        self.init_evaluation()

    def init_evaluation(self):
        self.state_value = defaultdict(int) # value of a state (0 if unknown)        
            
    def add_state(self, state):
        """Add a state if unknown."""
        state_code = self.environment.encode(state)
        if state_code not in self.state_value:
            self.state_value[state_code] = 0
        
    def get_states(self):
        """Get known states."""
        states = [self.environment.decode(state_code) for state_code in self.state_value]
        return states
    
    def is_known(self, state):
        """Check if some state is known."""
        return self.environment.encode(state) in self.state_value

    def get_values(self, states):
        """Get the value function of some states.""" 
        state_codes = [self.environment.encode(state) for state in states]
        values = [self.state_value[state_code] for state_code in state_codes]
        return values
    
    def improve_policy(self):
        """Improve policy using the current value function."""
        best_action = defaultdict(lambda: None)
        states = self.get_states()
        for state in states:
            actions = self.environment.get_actions(state)
            if len(actions):
                action_values = []
                for action in actions:
                    probs, states = self.environment.get_transition(state, action)
                    rewards = [self.environment.get_reward(state) for state in states]
                    values = self.get_values(states)
                    # expected value
                    value = np.sum(np.array(probs) * (np.array(rewards) + self.gamma * np.array(values)))
                    action_values.append(value)
                state_code = self.environment.encode(state)
                best_action[state_code] = np.argmax(action_values)
        def policy(state):
            actions = self.environment.get_actions(state)
            state_code = self.environment.encode(state)
            if best_action[state_code] is not None:
                action = actions[best_action[state_code]]
                return [1], [action]
            else:
                probs = np.ones(len(actions)) / len(actions)
                return probs, actions
        return policy    
        
        
class OnlineControl:
    """Online control. The agent interacts with the environment and learns the best policy."""
    
    def __init__(self, environment, gamma=1, alpha=0.1, eps=0.1, n_steps=1000):
        self.environment = environment
        self.is_game = hasattr(environment, "first_player")
        self.gamma = gamma # discount rate
        self.alpha = alpha # learning rate
        self.eps = eps # exploration rate
        self.n_steps = n_steps # maximum number of steps per episode
        self.init_evaluation()
                      
    def init_evaluation(self):
        self.state_action_value = defaultdict(lambda: defaultdict(int)) # value of a state-action pair (0 if unknown)  

    def add_state_action(self, state, action):
        """Add a state-action pair if unknown."""
        state_code = self.environment.encode(state)
        if state_code not in self.state_action_value:
            self.state_action_value[state_code][action] = 0
            
    def get_states(self):
        """Get known states."""
        states = [self.environment.decode(state_code) for state_code in self.state_action_value]
        return states
    
    def is_known(self, state):
        """Check if some state is known."""
        return self.environment.encode(state) in self.state_action_value
    
    def get_actions(self, state):
        if self.is_game:
            return self.environment.get_actions(state, 1)
        else:
            return self.environment.get_actions(state)
    
    def get_best_action(self, state):
        """Get the best action in some state.""" 
        actions = self.get_actions(state)
        state_code = self.environment.encode(state)
        values = np.array([self.state_action_value[state_code][action] for action in actions])
        i = np.random.choice(np.argwhere(values==np.max(values)).ravel())
        best_action = actions[i]
        return best_action

    def get_best_action_randomized(self, state):
        """Get the best action in some state, or a random state with probability epsilon.""" 
        if np.random.random() < self.eps:
            actions = self.get_actions(state)
            return actions[np.random.choice(len(actions))]
        else:
            return self.get_best_action(state)
        
    def get_policy(self):
        """Get the best known policy.""" 
        def policy(state):
            return [1], [self.get_best_action(state)]
        return policy
