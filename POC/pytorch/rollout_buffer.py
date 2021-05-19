import numpy as np

class RolloutBuffer():
    def __init__(self,N_STEPS=512, batch_size = 32, action_dim=2, state_dim=2):
        
        self.batch_size = batch_size

        self.states = np.zeros((N_STEPS, state_dim), dtype=np.float32)
        self.actions = np.zeros((N_STEPS, action_dim))
        self.rewards = np.zeros((N_STEPS, ), dtype=np.float32)
        self.dones = np.zeros((N_STEPS, ))
        
        self.log_probs = np.zeros((N_STEPS, ), dtype=np.float32)
        self.values = np.zeros((N_STEPS, ), dtype=np.float32)
        
        self.idx = 0 # Index of starting of batch
    
    def add(self, state, action, reward, done, log_prob):
        idx = self.idx
        # if idx >= 512:
        #     return
        # batch_size = self.batch_size
        # print(self.states.shape, state.shape)
        self.states[idx] = state#.copy()
        self.actions[idx] = action#.copy()
        self.rewards[idx] = reward
        self.dones[idx] = done
        self.log_probs[idx] = log_prob
        self.computed_values = False
        self.idx += 1
        
    def compute_values(self, last_value=0,gamma=0.99):

        n = self.idx

        running_sum = last_value
        for i in range(n-1,-1,-1):
            if self.dones[i]:
                running_sum = self.rewards[i]
            else:
                running_sum = self.rewards[i] + gamma*running_sum
            self.values[i] =  running_sum
        # self.compute_values = True
    def clear(self):
        self.idx = 0

    def __iter__(self):
        self.idx = 0
        return self
        
    def __next__(self):
        idx, batch_size = self.idx, self.batch_size
        if self.idx + self.batch_size <= len(self.states):
            s,a,v,l = self.states[idx:idx+batch_size],self.actions[idx:idx+batch_size],self.values[idx:idx+batch_size],self.log_probs[idx:idx+batch_size]
            self.idx+=1
            return s,a,v,l
        else:
            raise StopIteration