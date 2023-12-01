import numpy as np


policy_prob = 1/8 * np.ones((2,2,2))
print(policy_prob)
best_actions = np.argwhere(policy_prob == np.amax(policy_prob)).flatten().tolist()
print('1',np.amax(policy_prob))
print(np.argwhere(policy_prob == np.amax(policy_prob)))
print(best_actions)

