"""Pure PPO update on stored tensors. This module has no env/prior provider."""
import math
import torch
from torch import nn


def compute_gae(rewards, values, dones, last_value, gamma=.99, gae_lambda=.95):
    # V5 done includes task deadline failure: it is an actual terminal outcome.
    # Only a live rollout boundary bootstraps. Never cross a per-env reset.
    advantage, carry = torch.zeros_like(rewards), torch.zeros_like(last_value)
    for t in reversed(range(len(rewards))):
        next_value = last_value if t == len(rewards) - 1 else values[t + 1]
        keep = (~dones[t]).to(rewards.dtype)
        carry = rewards[t] + gamma * next_value * keep - values[t] + gamma * gae_lambda * keep * carry
        advantage[t] = carry
    return advantage, advantage + values


def make_optimizers(model, config):
    actor = model.actor_parameters()
    critic = list(model.critic.parameters())
    if {id(p) for p in actor} & {id(p) for p in critic}:
        raise ValueError('Actor/critic parameters must not be shared')
    return dict(actor=torch.optim.Adam(actor, lr=config.get('actor_lr', 3e-5)),
                critic=torch.optim.Adam(critic, lr=config.get('critic_lr', 3e-4)))


def ppo_update(model, batch, optimizers, config, update_actor=True):
    required = ('history', 'prior_delta', 'action', 'old_log_prob', 'advantage', 'returns')
    if any(key not in batch for key in required):
        raise ValueError('PPO minibatches require the rollout prior and full unclipped action')
    count = len(batch['history'])
    if not count or any(len(batch[key]) != count for key in required):
        raise ValueError('Rollout fields have inconsistent row counts')
    if any(not torch.isfinite(batch[key]).all() for key in required):
        raise ValueError('Nonfinite PPO rollout')
    actor_stats, critic_stats = [], []
    actor_stopped = not update_actor
    target_kl = config.get('target_kl', .015)
    clip_ratio = config.get('clip_ratio', .2)
    epochs = config.get('ppo_epochs', 4)
    for epoch in range(epochs):
        permutation = torch.randperm(count, device=batch['history'].device)
        for ids in permutation.split(config.get('batch_size', 4096)):
            history = batch['history'][ids]
            # Fresh critic can have large error; it has its own optimizer and clip.
            value_loss = .5 * (model.value(history) - batch['returns'][ids]).square().mean()
            optimizers['critic'].zero_grad(set_to_none=True)
            value_loss.backward()
            critic_norm = nn.utils.clip_grad_norm_(model.critic.parameters(), config.get('critic_grad_clip', 1.))
            if not torch.isfinite(critic_norm):
                raise RuntimeError('Nonfinite critic gradient')
            optimizers['critic'].step()
            critic_stats.append((float(value_loss.detach()), float(critic_norm)))
            if actor_stopped:
                continue
            # IMPORTANT: this is the prior saved with these old states, never a
            # new query against the live environment at PPO minibatch time.
            dist = model.distribution(history, batch['prior_delta'][ids])
            log_prob = dist.log_prob(batch['action'][ids]).sum(-1)
            log_ratio = log_prob - batch['old_log_prob'][ids]
            ratio = log_ratio.exp()
            kl = ((ratio - 1) - log_ratio).mean().detach()
            if not torch.isfinite(kl) or kl > target_kl:
                actor_stopped = True
                continue  # Critic updates still complete after policy early stop.
            advantage = batch['advantage'][ids]
            surrogate = torch.minimum(ratio * advantage, ratio.clamp(1-clip_ratio, 1+clip_ratio) * advantage)
            actor_loss = -surrogate.mean() - config.get('entropy_coefficient', 0.) * dist.entropy().sum(-1).mean()
            optimizers['actor'].zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_norm = nn.utils.clip_grad_norm_(model.actor_parameters(), config.get('actor_grad_clip', .5))
            if not torch.isfinite(actor_norm):
                raise RuntimeError('Nonfinite actor gradient')
            optimizers['actor'].step()
            model.project_logstd()
            actor_stats.append((float(actor_loss.detach()), float(kl), float(actor_norm)))
    def average(rows, index):
        return sum(row[index] for row in rows) / len(rows) if rows else None
    return dict(actor_loss=average(actor_stats,0), kl=average(actor_stats,1), actor_grad_norm=average(actor_stats,2),
                critic_loss=average(critic_stats,0), critic_grad_norm=average(critic_stats,1),
                actor_minibatches=len(actor_stats), critic_minibatches=len(critic_stats),
                actor_early_stopped=bool(update_actor and actor_stopped))
