import torch
from torch import nn
from torch.optim.adam import Adam


class PPO:
    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        e: float = 0.2,
        device: str = "cpu"
    ) -> None:
        self.device = device
        
        self.gamma = gamma
        self.lam = lam
        self.e = e
        
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.critic_opt = Adam(critic.parameters(), lr=critic_lr)
        self.actor_opt = Adam(actor.parameters(), lr=actor_lr)
    
    def compute_gae(self, rewards, values, dones):
        T = len(rewards)

        advantages = torch.zeros(T, device=self.device)
        gae = torch.zeros((), device=self.device)
        
        # calculating GAE for the whole rollout
        for t in reversed(range(T)):
            next_value = values[t + 1] if t + 1 < T else torch.zeros((), device=self.device)
            next_nonterminal = torch.tensor(1.0 - float(dones[t]), device=self.device)
            delta = rewards[t] + self.gamma * next_value * next_nonterminal - values[t]
            gae = delta + self.gamma * self.lam * next_nonterminal * gae
            advantages[t] = gae
            
        returns = advantages + values
        return advantages, returns
    
    def update_networks(
        self,
        obs_batch,
        actions_batch,
        old_log_probs,
        advantages,
        returns,
        epoch: int,
        minibatch_size: int,
    ):
        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        batch_size = obs_batch.shape[0]
        minibatch_size = minibatch_size
        
        for _ in range(epoch):
            indices = torch.randperm(batch_size)
            
            for start in range(0, batch_size, minibatch_size):
                idx = indices[start:start+minibatch_size]
                mb_obs = obs_batch[idx]
                mb_actions = actions_batch[idx]
                mb_old_log_probs = old_log_probs[idx]
                mb_advantages = advantages[idx]
                mb_returns = returns[idx]
                
                # get log_probs from the updated actor
                mean = self.actor(mb_obs)
                std = self.actor.log_std.clamp(min=-2.0).exp()
                dist = torch.distributions.Normal(mean, std)
                new_log_probs = dist.log_prob(mb_actions).sum(-1)
                entropy = dist.entropy().sum(-1).mean()
                
                # calculate actor loss
                ratio = (new_log_probs - mb_old_log_probs).exp()
                actor_loss = -torch.minimum(ratio * mb_advantages, torch.clamp(ratio, 1-self.e, 1+self.e) * mb_advantages).mean()
                actor_loss = actor_loss - 0.01 * entropy
                
                # calculate critic loss
                values = self.critic(mb_obs).squeeze()
                critic_loss = nn.functional.mse_loss(values, mb_returns)
                
                # update the networks
                self.actor_opt.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_opt.step()

                self.critic_opt.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_opt.step()