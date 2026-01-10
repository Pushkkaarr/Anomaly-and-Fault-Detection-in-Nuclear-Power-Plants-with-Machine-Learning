import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

# Initialize weights using Xavier/He initialization
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class QNetwork(nn.Module):
    """
    Critic Network (Q-function approximator)
    
    Architecture: Deeper network for complex nuclear dynamics
    Input: [state, action] → Output: Q-value
    """
    
    def __init__(self, state_dim, action_dim, hidden_dims=[512, 512, 256, 128]):
        super(QNetwork, self).__init__()
        
        # Build layers
        layers = []
        input_dim = state_dim + action_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))  # Stabilizes training
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))
        
        self.network = nn.Sequential(*layers)
        self.apply(weights_init_)
    
    def forward(self, state, action):
        """Forward pass through Q-network"""
        x = torch.cat([state, action], dim=1)
        return self.network(x)


class GaussianPolicy(nn.Module):
    """
    Actor Network (Stochastic Policy)
    
    Outputs: mean and log_std for Gaussian distribution over actions
    Uses tanh squashing for bounded continuous control
    """
    
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256], 
                 log_std_min=-20, log_std_max=2):
        super(GaussianPolicy, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Shared feature extraction
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            input_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Separate heads for mean and log_std
        self.mean_head = nn.Linear(input_dim, action_dim)
        self.log_std_head = nn.Linear(input_dim, action_dim)
        
        self.apply(weights_init_)
    
    def forward(self, state):
        """
        Forward pass
        Returns: mean and log_std of action distribution
        """
        features = self.feature_extractor(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        
        # Clamp log_std for numerical stability
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state):
        """
        Sample action from policy with reparameterization trick
        
        Returns:
            action: Sampled action (tanh squashed)
            log_prob: Log probability of action
            mean: Mean of action distribution (for evaluation)
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Reparameterization trick
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Differentiable sampling
        
        # Squash to [-1, 1] using tanh
        action = torch.tanh(x_t)
        
        # Compute log probability with tanh correction
        log_prob = normal.log_prob(x_t)
        
        # Enforcing Action Bound (correction term for tanh squashing)
        # This is critical for SAC! Without it, the policy diverges
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        # Mean action for evaluation (deterministic)
        mean_action = torch.tanh(mean)
        
        return action, log_prob, mean_action
    
    def get_action(self, state, deterministic=False):
        """
        Get action for inference (no gradient computation)
        
        Args:
            state: Current observation
            deterministic: If True, return mean action; else sample
        """
        with torch.no_grad():
            if deterministic:
                mean, _ = self.forward(state)
                return torch.tanh(mean)
            else:
                action, _, _ = self.sample(state)
                return action


class SACAgent:
    """
    Complete SAC Agent with Actor and Dual Critics
    
    Implements:
    - Soft Actor-Critic algorithm
    - Automatic entropy tuning
    - Target network soft updates
    - Gradient clipping for stability
    """
    
    def __init__(self, state_dim, action_dim, device='cuda', 
                 lr_actor=3e-4, lr_critic=3e-4, lr_alpha=3e-4,
                 gamma=0.99, tau=0.005, alpha=0.2, auto_entropy_tuning=True):
        
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.auto_entropy_tuning = auto_entropy_tuning
        
        # Networks
        self.actor = GaussianPolicy(state_dim, action_dim).to(device)
        
        # Dual Q-networks (standard in SAC)
        self.critic1 = QNetwork(state_dim, action_dim).to(device)
        self.critic2 = QNetwork(state_dim, action_dim).to(device)
        
        # Target networks (for stable training)
        self.critic1_target = QNetwork(state_dim, action_dim).to(device)
        self.critic2_target = QNetwork(state_dim, action_dim).to(device)
        
        # Initialize targets to match current networks
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=lr_critic)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=lr_critic)
        
        # Automatic entropy tuning
        if self.auto_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor([action_dim]).to(device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr_alpha)
        
        # Training statistics
        self.actor_losses = []
        self.critic_losses = []
        self.alpha_values = []
    
    def select_action(self, state, deterministic=False):
        """Select action for environment interaction"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor.get_action(state, deterministic)
        return action.cpu().numpy()[0]
    
    def update(self, replay_buffer, batch_size):
        """
        Update all networks using a batch from replay buffer
        
        Returns: Dictionary of training metrics
        """
        # Sample batch
        states, actions, rewards, next_states, dones = replay_buffer.sample_torch(
            batch_size, self.device, prioritized=True
        )
        
        # ====================================================================
        # UPDATE CRITICS (Q-functions)
        # ====================================================================
        with torch.no_grad():
            # Sample actions for next states from current policy
            next_actions, next_log_probs, _ = self.actor.sample(next_states)
            
            # Compute target Q-values (take minimum to reduce overestimation)
            target_q1 = self.critic1_target(next_states, next_actions)
            target_q2 = self.critic2_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            
            # Add entropy term (soft Q-learning)
            target_q = target_q - self.alpha * next_log_probs
            
            # Bellman backup
            target_value = rewards + (1 - dones) * self.gamma * target_q
        
        # Current Q-values
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        
        # Critic losses (MSE)
        critic1_loss = F.mse_loss(current_q1, target_value)
        critic2_loss = F.mse_loss(current_q2, target_value)
        
        # Update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
        self.critic2_optimizer.step()
        
        # ====================================================================
        # UPDATE ACTOR (Policy)
        # ====================================================================
        # Sample new actions from current policy
        new_actions, log_probs, _ = self.actor.sample(states)
        
        # Q-values for new actions
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        # Actor loss (maximize Q - α*entropy)
        actor_loss = (self.alpha * log_probs - q_new).mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # ====================================================================
        # UPDATE ENTROPY COEFFICIENT (if auto-tuning)
        # ====================================================================
        if self.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp().item()
        
        # ====================================================================
        # SOFT UPDATE TARGET NETWORKS
        # ====================================================================
        self._soft_update(self.critic1, self.critic1_target)
        self._soft_update(self.critic2, self.critic2_target)
        
        # Store metrics
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append((critic1_loss.item() + critic2_loss.item()) / 2)
        self.alpha_values.append(self.alpha)
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': (critic1_loss.item() + critic2_loss.item()) / 2,
            'alpha': self.alpha,
            'q_value': q_new.mean().item()
        }
    
    def _soft_update(self, source, target):
        """Soft update target network parameters"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
    
    def save(self, filepath):
        """Save model checkpoint"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic1_optimizer': self.critic1_optimizer.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict(),
            'alpha': self.alpha,
            'log_alpha': self.log_alpha if self.auto_entropy_tuning else None
        }, filepath)
    
    def load(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer'])
        self.alpha = checkpoint['alpha']
        if self.auto_entropy_tuning and checkpoint['log_alpha'] is not None:
            self.log_alpha.data = checkpoint['log_alpha']