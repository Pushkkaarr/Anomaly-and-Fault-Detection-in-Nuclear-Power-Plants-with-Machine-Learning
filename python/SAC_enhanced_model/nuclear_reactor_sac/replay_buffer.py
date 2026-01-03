import numpy as np
import torch

class ReplayBuffer:
    """
    Experience Replay Buffer for SAC
    
    Stores transitions: (state, action, reward, next_state, done)
    Implements prioritized sampling for nuclear safety-critical events
    """
    
    def __init__(self, state_dim, action_dim, max_size=200000):
        """
        Args:
            state_dim: Dimension of observation space
            action_dim: Dimension of action space
            max_size: Maximum buffer capacity
        """
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        # Storage arrays
        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)
        
        # Priority tracking for safety-critical experiences
        self.priorities = np.zeros(max_size, dtype=np.float32)
        self.priority_threshold = 0.8  # Top 20% get priority
        
    def add(self, state, action, reward, next_state, done):
        """Add a transition to the buffer"""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        # Assign priority (higher for safety-critical transitions)
        priority = self._calculate_priority(state, reward, done)
        self.priorities[self.ptr] = priority
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def _calculate_priority(self, state, reward, done):
        """
        Calculate experience priority for sampling
        High priority for:
        - Near-failure states
        - Large rewards (positive or negative)
        - Temperature approaching limits
        """
        priority = 0.1  # Base priority
        
        # Extract state features (assuming order from env)
        power = state[0]
        fuel_temp = state[2]
        
        # High priority for dangerous temperatures
        if fuel_temp > 1150.0:
            priority += 0.5
        
        # High priority for power excursions
        if power > 1.3 or power < 0.7:
            priority += 0.3
        
        # High priority for large rewards (learning signal)
        if abs(reward) > 50:
            priority += 0.4
        
        # Maximum priority for terminal states
        if done:
            priority += 0.8
        
        return min(priority, 1.0)
    
    def sample(self, batch_size, prioritized=True):
        """
        Sample a batch of transitions
        
        Args:
            batch_size: Number of transitions to sample
            prioritized: If True, use priority sampling for 50% of batch
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        if prioritized and self.size > batch_size * 2:
            # Sample 50% uniformly, 50% from high-priority experiences
            uniform_size = batch_size // 2
            priority_size = batch_size - uniform_size
            
            # Uniform sampling
            uniform_indices = np.random.randint(0, self.size, size=uniform_size)
            
            # Priority sampling (weighted by priority)
            priorities = self.priorities[:self.size]
            probabilities = priorities / priorities.sum()
            priority_indices = np.random.choice(
                self.size, 
                size=priority_size, 
                p=probabilities,
                replace=False
            )
            
            indices = np.concatenate([uniform_indices, priority_indices])
        else:
            # Pure uniform sampling
            indices = np.random.randint(0, self.size, size=batch_size)
        
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )
    
    def sample_torch(self, batch_size, device, prioritized=True):
        """Sample and return as PyTorch tensors"""
        states, actions, rewards, next_states, dones = self.sample(batch_size, prioritized)
        
        return (
            torch.FloatTensor(states).to(device),
            torch.FloatTensor(actions).to(device),
            torch.FloatTensor(rewards).to(device),
            torch.FloatTensor(next_states).to(device),
            torch.FloatTensor(dones).to(device)
        )
    
    def get_statistics(self):
        """Return buffer statistics for monitoring"""
        if self.size == 0:
            return {
                'size': 0,
                'avg_reward': 0.0,
                'high_priority_fraction': 0.0
            }
        
        valid_rewards = self.rewards[:self.size]
        valid_priorities = self.priorities[:self.size]
        
        return {
            'size': self.size,
            'avg_reward': float(np.mean(valid_rewards)),
            'std_reward': float(np.std(valid_rewards)),
            'min_reward': float(np.min(valid_rewards)),
            'max_reward': float(np.max(valid_rewards)),
            'high_priority_fraction': float(np.mean(valid_priorities > self.priority_threshold))
        }
    
    def __len__(self):
        return self.size