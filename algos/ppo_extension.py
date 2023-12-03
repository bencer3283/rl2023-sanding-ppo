# Extensions implemented:
#     1. decreasing noise variance: line 
#     2. decreasing entropy bonus in loss: line
#     3. clipped gradient: line
#     ~~4. clipped value loss: line~~
#     5. reward scaling
#     ~~6. observation scaling~~
#     7. reward clipping

from .agent_base import BaseAgent
from .ppo_utils import Policy
from .ppo_agent import PPOAgent

import utils.common_utils as cu
import torch
import numpy as np
import torch.nn.functional as F
import time
import gymnasium as gym

class PPOExtension(PPOAgent):
    def __init__(self, config=None):
        super(PPOExtension, self).__init__(config)
        self.env=config["env"]
        # self.env = gym.wrappers.NormalizeObservation(self.env)
        # self.env = gym.wrappers.TransformObservation(self.env, lambda reward: np.clip(reward, -10, 10))
        self.env = gym.wrappers.NormalizeReward(self.env)
        # self.env = gym.wrappers.TransformReward(self.env, lambda reward: np.clip(reward, -10, 10))
    
    def ppo_update(self, states, actions, rewards, next_states, dones, old_log_probs, targets):
        action_dists, values = self.policy(states)
        values = values.squeeze()
        new_action_probs = action_dists.log_prob(actions)
        ratio = torch.exp(new_action_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1-self.clip, 1+self.clip)

        advantages = targets - values
        advantages -= advantages.mean()
        advantages /= advantages.std()+1e-8
        advantages = advantages.detach()
        policy_objective = -torch.min(ratio*advantages, clipped_ratio*advantages)

        value_loss = F.smooth_l1_loss(values, targets, reduction="mean")

        policy_objective = policy_objective.mean()
        entropy = action_dists.entropy().mean()
        loss = policy_objective + value_loss - 0.001 * self.policy.actor_logstd * entropy # decreasing entropy bonus

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.05) # clip gradient norm 
        self.optimizer.step()
    
    def train_iteration(self,ratio_of_episodes):
        # Run actual training        
        reward_sum, episode_length, num_updates = 0, 0, 0
        done = False

        # Reset the environment and observe the initial state
        observation, _  = self.env.reset()

        while not done and episode_length < self.cfg.max_episode_steps:
            # Get action from the agent
            action, action_log_prob = self.get_action(observation)
            previous_observation = observation.copy()

            # Perform the action on the environment, get new state and reward
            observation, reward, done, _, _ = self.env.step(action)
            
            # Store action's outcome (so that the agent can improve its policy)
            self.store_outcome(previous_observation, action, observation,
                                reward, action_log_prob, done)

            # Store total episode reward
            reward_sum += reward
            episode_length += 1

            # Update the policy, if we have enough data
            if len(self.states) > self.cfg.min_update_samples:
                self.update_policy()
                num_updates += 1

                # Update policy randomness
                self.policy.set_logstd_ratio(ratio_of_episodes) # decreasing noise variance

        # Return stats of training
        update_info = {'episode_length': episode_length,
                    'ep_reward': reward_sum}
        return update_info