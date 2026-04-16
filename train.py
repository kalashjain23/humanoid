import mujoco
import torch
from torch import nn
import numpy as np
from torch.optim.adam import Adam

from model import Actor, Critic
from reward import upright, forward_velocity

        
def bent_pose(data):
    # bent left leg
    data.qpos[18] = -0.5
    data.qpos[21] = 0.5
    data.qpos[22] = -0.3
    
    # bent right leg
    data.qpos[24] = -0.5
    data.qpos[27] = 0.5
    data.qpos[28] = -0.3
    
    return data

def get_observation(model, data):
    sensors_data = np.array([])
    for i in range(model.nsensor):
        start = model.sensor_adr[i]
        end = model.sensor_adr[i]+model.sensor_dim[i]
        sensors_data = np.append(sensors_data, data.sensordata[start:end])
        
    obs = np.concatenate([sensors_data, data.qpos[7:], data.qvel[6:]])
    
    return obs

def reward_function(data):
    return forward_velocity(data) + (0.1 * upright(data))

def check_termination(data):
    return True if data.body("Waist").xpos[2] < 0.5 else False # initially waist is z=0.584

def compute_gae(rewards, values, dones, gamma, lam):
    T = len(rewards)
    advantages = torch.zeros(T)
    gae = 0.0
    
    # calculating GAE for the whole rollout
    for t in reversed(range(T)):
        next_value = values[t+1] if t+1 < T else 0.0
        next_nonterminal = 1.0 - float(dones[t])
        delta = rewards[t] + ((gamma * next_value) - values[t])
        gae = delta + gamma * lam * next_nonterminal * gae
        advantages[t] = gae
        
    returns = advantages + values
    return advantages, returns
        

def update_networks(
    actor: nn.Module,
    actor_opt: torch.optim.Optimizer,
    critic: nn.Module,
    critic_opt: torch.optim.Optimizer,
    obs_batch,
    actions_batch,
    old_log_probs,
    advantages,
    returns,
    e: float,
    epoch: int,
):
    # normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    for _ in range(epoch):
        # get log_probs from the updated actor
        mean = actor(obs_batch)
        std = actor.log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        new_log_probs = dist.log_prob(actions_batch).sum(-1)
        
        # calculate actor loss
        ratio = (new_log_probs - old_log_probs).exp()
        actor_loss = -torch.minimum(ratio * advantages, torch.clamp(ratio, 1-e, 1+e) * advantages).mean()
        
        # calculate critic loss
        values = critic(obs_batch).squeeze()
        critic_loss = nn.functional.mse_loss(values, returns)
        
        # update the networks
        actor_opt.zero_grad()
        actor_loss.backward()
        actor_opt.step()

        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()
    

def main():
    iterations = 1000
    rollout_length = 1000
    gamma = 0.99
    lam = 0.95
    e = 0.2
    epoch = 5
    
    model = mujoco.MjModel.from_xml_path("booster_t1/scene.xml")
    data = mujoco.MjData(model)
    
    actor = Actor()
    critic = Critic()
    all_rewards = []
    
    critic_opt = Adam(critic.parameters(), lr=0.001)
    actor_opt = Adam(actor.parameters(), lr=0.001)
        
    for i in range(iterations):
        # get humanoid in bent pose as the default pose
        mujoco.mj_resetData(model, data)
        data = bent_pose(data)
        mujoco.mj_forward(model, data)
        
        first_obs = get_observation(model, data)
        obs_history = [first_obs, first_obs, first_obs] # 168 dims for the networks
        rollouts = []
        
        for _ in range(rollout_length):
            stacked_obs = torch.tensor(np.concatenate(obs_history), dtype=torch.float32)
            
            # calculating actor log probs
            mean = actor(stacked_obs)
            std = actor.log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample().detach()
            log_prob = dist.log_prob(action).sum(-1).detach()
            
            # applying the action
            data.ctrl[:] = action.detach().numpy()
            mujoco.mj_step(model, data)
            
            # calculating the reward (reward function) and value (by critic)
            reward = torch.tensor(reward_function(data), dtype=torch.float32)
            value = critic(stacked_obs).squeeze().detach()
            done = check_termination(data)
            
            # update history
            new_obs = get_observation(model, data)
            obs_history.pop(0)
            obs_history.append(new_obs)
            
            if done:
                mujoco.mj_resetData(model, data)
                data = bent_pose(data)
                mujoco.mj_forward(model, data)
                
                # reset obs history
                first_obs = get_observation(model, data)
                obs_history = [first_obs, first_obs, first_obs]
                
            rollouts.append((stacked_obs, action, log_prob, reward, value, done))
                
        obs_batch = torch.stack([r[0] for r in rollouts])
        actions_batch = torch.stack([r[1] for r in rollouts])
        old_log_probs = torch.stack([r[2] for r in rollouts])
        rewards = torch.stack([r[3] for r in rollouts])
        values = torch.stack([r[4] for r in rollouts])
        dones = [r[5] for r in rollouts]
        
        advantages, returns = compute_gae(rewards, values, dones, gamma, lam)
        update_networks(actor, actor_opt, critic, critic_opt, obs_batch, actions_batch, old_log_probs, advantages, returns, e, epoch)
        
        all_rewards.append(rewards.mean().item())
        print(f"iter {i}, mean_reward={rewards.mean():.3f}")
        
    torch.save({
        'actor': actor.state_dict(),
        'critic': critic.state_dict(),
        'rewards': all_rewards,
    }, 'checkpoint.pt')


if __name__ == "__main__":
    main()
