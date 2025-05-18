"""
Deep Q-Learning implementation.
"""

from typing import Any, Dict, List, Tuple

import os

import gymnasium as gym
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from numpy import floating
from omegaconf import DictConfig
from rl_exercises.agent import AbstractAgent
from rl_exercises.week_4.buffers import ReplayBuffer
from rl_exercises.week_4.networks import QNetwork


def set_seed(env: gym.Env, seed: int = 0) -> None:
    """
    Seed Python, NumPy, PyTorch and the Gym environment for reproducibility.

    Parameters
    ----------
    env : gym.Env
        The Gym environment to seed.
    seed : int
        Random seed.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.reset(seed=seed)
    # some spaces also support .seed()
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    if hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)


class DQNAgent(AbstractAgent):
    """
    Deep Q-Learning agent with ε-greedy policy and target network.

    Derives from AbstractAgent by implementing:
      - predict_action
      - save / load
      - update_agent
    """

    def __init__(
        self,
        env: gym.Env,
        buffer_capacity: int = 10000,
        batch_size: int = 32,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_final: float = 0.01,
        epsilon_decay: int = 500,
        target_update_freq: int = 1000,
        seed: int = 0,
    ) -> None:
        """
        Initialize replay buffer, Q-networks, optimizer, and hyperparameters.

        Parameters
        ----------
        env : gym.Env
            The Gym environment.
        buffer_capacity : int
            Max experiences stored.
        batch_size : int
            Mini-batch size for updates.
        lr : float
            Learning rate.
        gamma : float
            Discount factor.
        epsilon_start : float
            Initial ε for exploration.
        epsilon_final : float
            Final ε.
        epsilon_decay : int
            Exponential decay parameter.
        target_update_freq : int
            How many updates between target-network syncs.
        seed : int
            RNG seed.
        """
        super().__init__(
            env,
            buffer_capacity,
            batch_size,
            lr,
            gamma,
            epsilon_start,
            epsilon_final,
            epsilon_decay,
            target_update_freq,
            seed,
        )
        self.env = env
        set_seed(env, seed)

        obs_dim = env.observation_space.shape[0]
        n_actions = env.action_space.n

        # main Q-network and frozen target
        self.q = QNetwork(obs_dim, n_actions)
        self.target_q = QNetwork(obs_dim, n_actions)
        self.target_q.load_state_dict(self.q.state_dict())

        self.optimizer = optim.Adam(self.q.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)

        # hyperparams
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq

        self.total_steps = 0  # for ε decay and target sync

    def epsilon(self) -> float:
        """
        Compute current ε by exponential decay.

        Returns
        -------
        float
            Exploration rate.
        """

        # ε = ε_final + (ε_start - ε_final) * exp(-total_steps / ε_decay)
        return self.epsilon_final + (self.epsilon_start - self.epsilon_final) * np.exp(
            -self.total_steps / self.epsilon_decay
        )

    def predict_action(
        self, state: np.ndarray, evaluate: bool = False
    ) -> Tuple[int, Dict]:
        """
        Choose action via ε-greedy (or purely greedy in eval mode).

        Parameters
        ----------
        state : np.ndarray
            Current observation.
        info : dict
            Gym info dict (unused here).
        evaluate : bool
            If True, always pick argmax(Q).

        Returns
        -------
        action : int
        info_out : dict
            Empty dict (compatible with interface).
        """
        if evaluate:
            with torch.no_grad():
                qvals = self.q(torch.tensor(state, dtype=torch.float32))  # noqa: F841

            action = int(torch.argmax(qvals).item())
        else:
            if np.random.rand() < self.epsilon():
                action = self.env.action_space.sample()
            else:
                qvals = self.q(torch.tensor(state, dtype=torch.float32))
                action = int(torch.argmax(qvals).item())

        return action

    def save(self, path: str) -> None:
        """
        Save model & optimizer state to disk.

        Parameters
        ----------
        path : str
            File path.
        """
        torch.save(
            {
                "parameters": self.q.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        """
        Load model & optimizer state from disk.

        Parameters
        ----------
        path : str
            File path.
        """
        checkpoint = torch.load(path)
        self.q.load_state_dict(checkpoint["parameters"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

    def update_agent(
        self, training_batch: List[Tuple[Any, Any, float, Any, bool, Dict]]
    ) -> float:
        """
        Perform one gradient update on a batch of transitions.

        Parameters
        ----------
        training_batch : list of transitions
            Each is (state, action, reward, next_state, done, info).

        Returns
        -------
        loss_val : float
            MSE loss value.
        """
        # unpack
        states, actions, rewards, next_states, dones, _ = zip(*training_batch)  # noqa: F841
        s = torch.tensor(np.array(states), dtype=torch.float32)  # noqa: F841
        a = torch.tensor(np.array(actions), dtype=torch.int64).unsqueeze(1)  # noqa: F841
        r = torch.tensor(np.array(rewards), dtype=torch.float32)  # noqa: F841
        s_next = torch.tensor(np.array(next_states), dtype=torch.float32)  # noqa: F841
        mask = torch.tensor(np.array(dones), dtype=torch.float32)  # noqa: F841

        ## Gather Q(s,a)
        pred = self.q(s).gather(1, a)

        # compute TD target with frozen network
        with torch.no_grad():
            next_qvals = self.target_q(s_next)
            max_next_qvals = next_qvals.max(dim=1, keepdim=True)[0]  # [batch_size, 1]

            target = (
                r.unsqueeze(1) + (1 - mask.unsqueeze(1)) * self.gamma * max_next_qvals
            )

        loss = nn.MSELoss()(pred, target)

        # gradient step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # occasionally sync target network
        if self.total_steps % self.target_update_freq == 0:
            self.target_q.load_state_dict(self.q.state_dict())

        self.total_steps += 1
        return float(loss.item())

    def train(
        self, num_frames: int, eval_interval: int = 1000
    ) -> list[tuple[int, floating[Any]]]:
        """
        Run a training loop for a fixed number of frames.

        Parameters
        ----------
        num_frames : int
            Total environment steps.
        eval_interval : int
            Every this many episodes, print average reward.
        """
        state, _ = self.env.reset()
        ep_reward = 0.0
        recent_rewards: List[float] = []
        rewards_avg = []

        for frame in range(1, num_frames + 1):
            action = self.predict_action(state)
            next_state, reward, done, truncated, _ = self.env.step(action)

            # store and step
            self.buffer.add(state, action, reward, next_state, done or truncated, {})
            state = next_state
            ep_reward += reward

            # update if ready
            if len(self.buffer) >= self.batch_size:
                batch = self.buffer.sample(self.batch_size)
                _ = self.update_agent(batch)

            if done or truncated:
                state, _ = self.env.reset()
                recent_rewards.append(ep_reward)
                ep_reward = 0.0
                # logging
                if len(recent_rewards) % 10 == 0:
                    avg = np.mean(recent_rewards[-10:])
                    print(
                        f"Frame {frame}, AvgReward(10): {avg:.2f}, ε={self.epsilon():.3f}"
                    )
                    rewards_avg.append((frame, avg))

        print("Training complete.")

        return rewards_avg


@hydra.main(config_path="../configs/agent/", config_name="dqn", version_base="1.1")
def main(cfg: DictConfig):
    all_curves = []
    arch_str = None
    buf_str = None
    bs_str = None

    for seed in cfg.seed:
        print(f"\n RUN seed={seed}")
        env = gym.make(cfg.env.name)
        set_seed(env, seed)

        agent = DQNAgent(env, **cfg.agent)

        if arch_str is None:
            layers = [m for m in agent.q.net if isinstance(m, nn.Linear)]
            hidden_sizes = [layer.out_features for layer in layers[:-1]]
            arch_str = "x".join(str(h) for h in hidden_sizes)
            buf_str = f"buf{agent.buffer.capacity // 1000}k"
            bs_str = f"bs{agent.batch_size}"

        curve = agent.train(cfg.train.num_frames, cfg.train.eval_interval)
        all_curves.append(curve)

    min_len = min(len(curve) for curve in all_curves)
    frames = [pt[0] for pt in all_curves[0][:min_len]]

    # compute mean and standard deviation
    means = []
    stds = []
    for i in range(min_len):
        vals = [curve[i][1] for curve in all_curves]
        means.append(np.mean(vals))
        stds.append(np.std(vals))

    # plots
    os.makedirs("../../../plots", exist_ok=True)
    plt.plot(frames, means, label="mean over seeds")
    plt.fill_between(
        frames,
        np.array(means) - np.array(stds),
        np.array(means) + np.array(stds),
        alpha=0.3,
        label="standard deviation",
    )
    plt.xlabel("Frame")
    plt.ylabel("Average Reward (10 episode window)")
    plt.title(f"DQN on {cfg.env.name}")
    plt.legend()
    plt.grid(True)

    info_text = (
        f"Arch: {arch_str}\n"
        f"Buffer: {agent.buffer.capacity}\n"
        f"Batch size: {agent.batch_size}\n"
    )
    plt.text(
        0.05,
        0.95,
        info_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.8),
    )

    filename = f"arch{arch_str}_{buf_str}_{bs_str}.png"
    filepath = os.path.join("../../../plots", filename)
    plt.savefig(filepath)
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
