"""Trainable correction around a recorded, frozen terrain action prior."""
import math
import torch
from torch import nn
from torch.distributions import Normal


class WarmResidualPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim=69, width=256, correction_limit=.08,
                 initial_std=.012, min_std=.006, max_std=.04):
        super().__init__()
        if not 0 < min_std <= initial_std <= max_std or correction_limit <= 0:
            raise ValueError('Invalid correction/noise bounds')
        self.obs_dim, self.action_dim = obs_dim, action_dim
        self.correction_limit = float(correction_limit)
        self.min_std, self.max_std = float(min_std), float(max_std)
        self.model_config = dict(obs_dim=obs_dim, action_dim=action_dim, width=width,
            correction_limit=correction_limit, initial_std=initial_std, min_std=min_std, max_std=max_std)
        self.actor = nn.Sequential(nn.Linear(obs_dim, width), nn.ELU(), nn.Linear(width, width),
                                   nn.ELU(), nn.Linear(width, action_dim))
        self.critic = nn.Sequential(nn.Linear(obs_dim, width), nn.ELU(), nn.Linear(width, width),
                                    nn.ELU(), nn.Linear(width, 1))
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)
        nn.init.zeros_(self.critic[-1].weight)
        nn.init.zeros_(self.critic[-1].bias)
        self.logstd = nn.Parameter(torch.full((action_dim,), math.log(initial_std)))

    def actor_parameters(self):
        return [*self.actor.parameters(), self.logstd]

    def correction(self, history):
        return self.correction_limit * self.actor(history).tanh()

    def distribution(self, history, prior_delta):
        if prior_delta.shape != (len(history), self.action_dim):
            raise ValueError('Each sample requires its own recorded action prior')
        mean = prior_delta.detach() + self.correction(history)
        return Normal(mean, self.logstd.clamp(math.log(self.min_std), math.log(self.max_std)).exp())

    def value(self, history):
        return self.critic(history).squeeze(-1)

    def forward(self, history, prior_delta):
        return self.distribution(history, prior_delta), self.value(history)

    @torch.no_grad()
    def project_logstd(self):
        self.logstd.clamp_(math.log(self.min_std), math.log(self.max_std))


def replace_prior_penalty(reward, executed_residual, executed_total, prior_total):
    """Only replace V5's -.5*residual²; preserve all other reward terms.

    The correction is measured AFTER the exact actuator composition/clipping.
    This cannot penalize the frozen terrain prior as if it were learned effort.
    """
    actual_correction = executed_total - prior_total
    old_penalty = .5 * executed_residual.square().mean(-1)
    new_penalty = .5 * actual_correction.square().mean(-1)
    return reward + old_penalty - new_penalty, actual_correction
