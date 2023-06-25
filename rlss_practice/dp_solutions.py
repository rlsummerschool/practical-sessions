"""Solutions of the first practical session."""

import numpy as np
from gymnasium import Env

from rlss_practice.environments import MinigridBase


class PolicyIteration:
  """
  Implements policy iteration
  """
  def __init__(self, env: MinigridBase, discount_factor: float, initial_policy = None):
    # Store
    self.env = env
    self.states = self.env.states
    self.n_states = len(self.states)
    self.actions = self.env.actions
    self.n_actions = len(self.actions)
    self.gamma = discount_factor
    self.policy = initial_policy

    # Default policy
    if self.policy is None:
      np.random.seed(4)
      self.policy = {state: np.random.randint(0, self.n_actions-1) for state in self.states}

    self.policy_stable = False
    self.V = {state: 0.0 for state in env.states}
    self.V_logs = []


  def _evaluate_policy(self):
    """
    Given 'policy' π_{k} compute V^{π_{k}}. Let,

      A = (I - \gamma p_{π_{k}})
      b = r_{π_{k}}
      x = V^{π_{k}}

    solve the system of linear equations Ax=b

    :return x: a |S|x1 array
    """
    A = np.eye(self.n_states, self.n_states) - self.gamma * self.get_p_pi()
    b = self.get_r_pi()
    return np.linalg.solve(A, b)


  def evaluate_policy(self):
    """
    Collect state values in dict
    """
    # Value
    V_array = self._evaluate_policy()

    self.V = {state: V_array[i].item() for i, state in enumerate(self.states)}
    self.V_logs.append(self.V.copy())


  def get_policy(self):
    """
    Get the greedy policy:

      π_{k+1}(s) = argmax_{a\in A} Q^{\pi_{k}}(s,a)

    where

      Q^{\pi_{k}}(s,a) = r(s,a) + gamma * <P(.|s,a),V^{\pi_{k}}>

    assign new policy to self.policy
    update self.policy_stable
    """
    self.policy_stable = True

    for state in self.states:
      max_Q_value = self.get_expected_update(state, self.policy[state])

      for action in self.actions:
        Q_value = self.get_expected_update(state, action)

        if action != self.policy[state] and max_Q_value < Q_value:
          self.policy[state] = action
          max_Q_value = Q_value
          self.policy_stable = False
    return


  def get_expected_update(self, state, action):
    """
    Compute:

      Q(s,a) = r(s,a) + gamma <p(.|s,a),V>

    :param state
    :param action

    :return Q(s,a): float
    """
    value  = self.env.R[state][action]

    for snext in self.states:
      value += self.gamma * self.env.T[state][action][snext] * self.V[snext]

    return value


  def get_p_pi(self):
    """
    Given π_{k}, compute p_{π_{k}}
    """
    p_pi = np.zeros((self.n_states,self.n_states))

    for s_index, s in enumerate(self.states):
      for snext_index, snext in enumerate(self.states):
        p_pi[s_index, snext_index] = self.env.T[s][self.policy[s]][snext]

    return p_pi


  def get_r_pi(self):
    """
    Given π_{k}, compute r_{π_{k}}
    """
    r_pi = np.zeros((self.n_states,1))

    for i, state in enumerate(self.states):
      r_pi[i][0] = self.env.R[state][self.policy[state]]

    return r_pi


class ModifiedPolicyIteration(PolicyIteration):
  """
  Implements policy iteration with truncated policy evaluation
  """
  def __init__(self,
              env: Env,
              discount_factor: float,
              theta: float,
              initial_policy = None):

    super().__init__(env, discount_factor, initial_policy)
    self.theta = theta
    self.V = {state: 0.0 for state in env.states}


  def _evaluate_policy(self):
    """
    Given 'policy' π_{k},
    Starting from a previous guess 'V_{k-1}',
    iteratively estimate V^{π_{k}} as T^{π_{k}}V_{k-1}.

    assign new state values to self.V
    """
    max_value_gap = np.inf

    while max_value_gap > self.theta:
      max_value_gap = 0

      for state in self.env.states:
        prev_statevalue = self.V[state]
        self.V[state] = self.get_expected_update(state, self.policy[state])
        max_value_gap = max(max_value_gap, np.abs(prev_statevalue - self.V[state]))


  def evaluate_policy(self):
    """
    Collect state values in dict
    """
    self._evaluate_policy()
    self.V_logs.append(self.V.copy())


class ValueIteration:
  """
  Implements value iteration
  """
  def __init__(self,
              env: Env,
              discount_factor: float,
              epsilon: float,
              num_iterations = None,
  ):
    self.env = env
    self.states = self.env.states
    self.n_states = len(self.states)
    self.actions = self.env.actions
    self.n_actions = len(self.actions)
    self.gamma = discount_factor
    self.K = num_iterations
    self.V_logs = []

    if self.K is None:
      self.K = math.ceil(np.log((2*self.gamma)/(epsilon*(1-self.gamma)**2))/(1-self.gamma))

    self.policy = {state: 0 for state in self.env.states}
    self.V = {state: 0 for state in self.env.states}
  
  def _estimate_vstar(self):
    for k in range(self.K):
      for state in self.env.states:
        self.V[state] = max(map(lambda action: self.get_expected_update(state, action), self.env.actions))

  def estimate_vstar(self):
    """
    Starting from an initial guess 'V',
    iteratively estimate V^{π*} as T*V

    Assign new values to self.V
    """
    self._evaluate_policy()
    self.V_logs.append(self.V.copy())


  def get_policy(self):
    """
    Get the greedy policy:
      π(s) = argmax_{a\in A} Q(s,a)
    where
      Q(s,a) = r(s,a) + gamma*<P(.|s,a),v>

    assign new policy to self.policy
    """
    # Same as for PI
    return PolicyIteration.get_policy(self)


  def get_expected_update(self, state, action):
    """
    Compute:

      Q(s,a) = r(s,a) + gamma <p(.|s,a),V>

    :param state
    :param action

    :return Q(s,a): float
    """
    # Same as for PI
    return PolicyIteration.get_expected_update(self, state, action)
