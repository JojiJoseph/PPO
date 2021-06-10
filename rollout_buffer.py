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
        self.advantages = np.zeros((N_STEPS, ), dtype=np.float32)
        self.returns = np.zeros((N_STEPS, ), dtype=np.float32)
        
        self.idx = 0 # Index of starting of batch
    
    def add(self, state, action, reward, done, log_prob, value):
        idx = self.idx
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = done
        self.log_probs[idx] = log_prob
        self.values[idx] = value
        self.computed_values = False
        self.idx += 1
        
    def compute_values(self, last_value=0,gamma=0.99, lda=1.0):
        n = self.idx
        prev_adv = 0
        for i in range(n-1,-1,-1):
            if self.dones[i]:
                delta = self.rewards[i] - self.values[i]
            else:
                delta = self.rewards[i] + gamma*last_value - self.values[i]
            adv = delta + lda*gamma*prev_adv
            prev_adv = adv
            last_value = self.values[i]
            self.advantages[i] = adv
            self.returns[i] = adv + self.values[i]
    def clear(self):
        self.idx = 0

    def __iter__(self):
        self.idx = 0
        return self
        
    def __next__(self):
        idx, batch_size = self.idx, self.batch_size
        if self.idx + self.batch_size <= len(self.states):
            s,a,adv,ret,l = self.states[idx:idx+batch_size],self.actions[idx:idx+batch_size],self.advantages[idx:idx+batch_size], self.returns[idx:idx+batch_size],self.log_probs[idx:idx+batch_size]
            self.idx+=self.batch_size
            return s,a,adv, ret,l
        else:
            raise StopIteration

class RolloutBufferMultiEnv():
    def __init__(self,n_steps=512, n_envs = 1, batch_size = 32, action_dim=2, state_dim=2):
        
        self.batch_size = batch_size
        n_rows = n_steps #// n_envs
        self.n_batch_rows = batch_size // n_envs
        self.n_envs = n_envs
        self.action_dim = action_dim
        self.state_dim = state_dim

        self.states = np.zeros((n_rows, n_envs, state_dim), dtype=np.float32)
        self.actions = np.zeros((n_rows, n_envs, action_dim))
        self.rewards = np.zeros((n_rows, n_envs,), dtype=np.float32)
        self.dones = np.zeros((n_rows, n_envs, ))
        
        self.log_probs = np.zeros((n_rows,n_envs, ), dtype=np.float32)
        self.values = np.zeros((n_rows,n_envs, ), dtype=np.float32)
        self.advantages = np.zeros((n_rows,n_envs, ), dtype=np.float32)
        self.returns = np.zeros((n_rows,n_envs, ), dtype=np.float32)
        
        self.idx = 0 # Index of starting of batch
    
    def add(self, state, action, reward, done, log_prob, value):
        idx = self.idx
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = done
        self.log_probs[idx] = log_prob
        self.values[idx] = value
        self.computed_values = False
        self.idx += 1
        
    def compute_values(self, last_value=0,gamma=0.99, lda=1.0):
        n = self.idx
        prev_adv = 0
        last_value = last_value.reshape((-1,))
        for i in range(n-1,-1,-1):
            delta = self.rewards[i] + gamma*last_value*(1-self.dones[i]) - self.values[i]
            adv = delta + lda*gamma*(1-self.dones[i])*prev_adv
            prev_adv = adv
            last_value = self.values[i]
            self.advantages[i] = adv
            self.returns[i] = adv + self.values[i]
    
    def clear(self):
        self.idx = 0

    def __iter__(self):
        self.idx = 0
        return self
        
    def __next__(self):
        idx, batch_size , batch_rows= self.idx, self.batch_size, self.n_batch_rows
        if self.idx + self.n_batch_rows<= len(self.states):
            s,a,adv,ret,l = self.states[idx:idx+batch_rows],self.actions[idx:idx+batch_rows],self.advantages[idx:idx+batch_rows], self.returns[idx:idx+batch_rows],self.log_probs[idx:idx+batch_rows]
            self.idx+=batch_rows
            s = s.reshape((-1,self.state_dim))
            a = a.reshape((-1,self.action_dim))
            adv = adv.reshape((-1,))
            ret = adv.reshape((-1,))
            l = l.reshape((-1,))
            return s,a,adv,ret,l
        else:
            raise StopIteration