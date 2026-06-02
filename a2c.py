import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim
import numpy as np
from collections import deque
import gymnasium as gym
from dataclasses import dataclass

from a2c_eval import show_video, evaluate_agent, plot_learning_and_scores_curves


@dataclass
class Args:
    env_id:                str   = "Acrobot-v1"
    n_updates:             int   = 3000 
    episodes_per_update:   int   = 8 
    n_evaluation_episodes: int   = 100
    max_t:                 int   = 500
    gamma:                 float = 0.99
    gae_lambda:            float = 0.95
    lr_actor:              float = 7e-4    # actor wants a SMALL lr (policy collapses if too fast)
    lr_critic:             float = 3e-3    # critic wants a LARGER lr (must fit big-magnitude returns)
    h_size:                int   = 128
    print_every:           int   = 20
    entropy_coef:          float = 0.01
    max_grad_norm:         float = 0.5
    normalize_adv:         bool  = True    # applied over the FULL batch, never per episode


# ── Networks ─────────────────────────────────────────────────────────────────

def _init_weights(module: nn.Module, gain: float = 5/3) -> None:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.zeros_(module.bias)


def _mlp(s_size: int, h_size: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(s_size, h_size), nn.Tanh(),
        nn.Linear(h_size, h_size), nn.Tanh(),
    )


class Actor(nn.Module):
    def __init__(self, s_size: int, a_size: int, h_size: int) -> None:
        super().__init__()
        self.body = _mlp(s_size, h_size)
        self.head = nn.Linear(h_size, a_size)
        self.body.apply(lambda m: _init_weights(m, gain=5/3))
        _init_weights(self.head, gain=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.head(self.body(x)), dim=-1)


class Critic(nn.Module):
    def __init__(self, s_size: int, h_size: int) -> None:
        super().__init__()
        self.body = _mlp(s_size, h_size)
        self.head = nn.Linear(h_size, 1)
        self.body.apply(lambda m: _init_weights(m, gain=5/3))
        _init_weights(self.head, gain=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x)).squeeze(-1)


class ActorCritic(nn.Module):

    def __init__(self, s_size: int, a_size: int, h_size: int) -> None:
        super().__init__()
        self.actor  = Actor(s_size, a_size, h_size)
        self.critic = Critic(s_size, h_size)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.actor(x), self.critic(x)

    def act(self, state: np.ndarray) -> tuple[int, torch.Tensor]:
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = self.actor(state_t)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


# ── GAE ──────────────────────────────────────────────────────────────────────

def compute_gae(
    rewards:    list[float],
    values:     list[float],
    last_value: float,
    gamma:      float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    
    T = len(rewards)
    values_ext = values + [last_value]
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        error       = rewards[t] + gamma * values_ext[t + 1] - values_ext[t]
        gae           = error + gamma * gae_lambda * gae
        advantages[t] = gae
    returns = advantages + np.asarray(values, dtype=np.float32)
    return advantages, returns


# ── Rollout collection ─────────────────────────────────────────────────────────

def collect_episode(policy: ActorCritic, env: gym.Env, max_t: int, device):
    states_buf:  list[np.ndarray] = []
    actions_buf: list[int]        = []
    rewards_buf: list[float]      = []
    terminated = truncated = False
    state, _ = env.reset()

    with torch.no_grad():
        for _ in range(max_t):
            states_buf.append(state.copy())
            state_t  = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            probs    = policy.actor(state_t)
            action   = Categorical(probs).sample().item()
            actions_buf.append(action)
            state, reward, terminated, truncated, _ = env.step(action)
            rewards_buf.append(reward)
            if terminated or truncated:
                break

        if truncated and not terminated:
            last_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            last_value = policy.critic(last_t).item()
        else:
            last_value = 0.0

    return states_buf, actions_buf, rewards_buf, last_value


# ── Training loop ──────────────────────────────────────────────────────────────

def a2c(
    policy:               ActorCritic,
    actor_optimizer:      optim.Optimizer,
    critic_optimizer:     optim.Optimizer,
    env:                  gym.Env,
    n_updates:            int,
    episodes_per_update:  int,
    max_t:                int,
    gamma:                float,
    gae_lambda:           float,
    print_every:          int,
    entropy_coef:         float,
    max_grad_norm:        float,
    normalize_adv:        bool,
) -> tuple[list[float], list[float]]:

    device = policy.device
    scores_deque = deque(maxlen=print_every * episodes_per_update)
    loss_deque   = deque(maxlen=print_every)
    scores, actor_losses = [], []

    for i_update in range(1, n_updates + 1):
        batch_states, batch_actions, batch_adv, batch_ret = [], [], [], []
        for _ in range(episodes_per_update):
            s_buf, a_buf, r_buf, last_value = collect_episode(policy, env, max_t, device)
            scores_deque.append(sum(r_buf))
            scores.append(sum(r_buf))

            with torch.no_grad():
                s_t  = torch.tensor(np.array(s_buf), dtype=torch.float32).to(device)
                v    = policy.critic(s_t)
            adv, ret = compute_gae(r_buf, v.tolist(), last_value, gamma, gae_lambda)

            batch_states.append(np.array(s_buf, dtype=np.float32))
            batch_actions.append(np.array(a_buf, dtype=np.int64))
            batch_adv.append(adv)
            batch_ret.append(ret)

        # ── Assemble the full batch ───────────────────────────────────────
        states_t   = torch.tensor(np.concatenate(batch_states), dtype=torch.float32).to(device)
        actions_t  = torch.tensor(np.concatenate(batch_actions), dtype=torch.long).to(device)
        advantages = torch.tensor(np.concatenate(batch_adv), dtype=torch.float32).to(device)
        returns    = torch.tensor(np.concatenate(batch_ret), dtype=torch.float32).to(device)

        # Normalize advantages over the WHOLE batch (valid; never per single episode).
        if normalize_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ── Actor update ──────────────────────────────────────────────────
        probs      = policy.actor(states_t)
        dist       = Categorical(probs)
        log_probs  = dist.log_prob(actions_t)
        entropy    = dist.entropy().mean()
        actor_loss = -(log_probs * advantages).mean() - entropy_coef * entropy

        actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(policy.actor.parameters(), max_norm=max_grad_norm)
        actor_optimizer.step()

        # ── Critic update ───────────────────────────────────────────────
        values_b    = policy.critic(states_t)
        critic_loss = F.mse_loss(values_b, returns)

        critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(policy.critic.parameters(), max_norm=max_grad_norm)
        critic_optimizer.step()

        loss_deque.append(actor_loss.item())
        actor_losses.append(actor_loss.item())

        if i_update % print_every == 0:
            print(
                f"Update {i_update:>5} ({i_update * episodes_per_update:>6} eps)"
                f" | Avg Score:  {np.mean(scores_deque):>7.1f}"
                f" | Actor Loss: {np.mean(loss_deque):>7.3f}"
                f" | Entropy:    {entropy.item():.4f}"
                f" | V(s):       {values_b.mean().item():>7.1f}"
                f" | V target:   {returns.mean().item():>7.1f}"
            )

    return scores, actor_losses

if __name__ == "__main__":
    args   = Args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    env    = gym.make(args.env_id, render_mode="rgb_array")
    s_size = env.observation_space.shape[0]
    a_size = env.action_space.n
    print(f"env: {args.env_id} | obs: {s_size} | act: {a_size} | device: {device}")

    policy           = ActorCritic(s_size, a_size, args.h_size).to(device)
    actor_optimizer  = optim.Adam(policy.actor.parameters(),  lr=args.lr_actor)
    critic_optimizer = optim.Adam(policy.critic.parameters(), lr=args.lr_critic)

    show_video(gym.make(args.env_id, render_mode="rgb_array"),
               policy, filename="acrobot_start.gif", fps=30)

    scores, losses = a2c(
        policy, actor_optimizer, critic_optimizer, env,
        args.n_updates, args.episodes_per_update, args.max_t,
        args.gamma, args.gae_lambda,
        args.print_every, args.entropy_coef,
        args.max_grad_norm, args.normalize_adv,
    )

    mean_reward, std_reward = evaluate_agent(
        gym.make(args.env_id, render_mode="rgb_array"),
        args.max_t, args.n_evaluation_episodes, policy
    )
    print(f"Mean reward ({args.n_evaluation_episodes} eps): {mean_reward:.2f} ± {std_reward:.2f}")

    show_video(gym.make(args.env_id, render_mode="rgb_array"),
               policy, filename="acrobot_end.gif", fps=30)

    plot_learning_and_scores_curves(scores, losses)