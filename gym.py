"""
Minimal Walker2d timing diagnostic for ReinforceAgent.
Tracks time per episode and per forward pass, then plots them.
"""

import time
import gymnasium as gym
import matplotlib.pyplot as plt

from src.agents.ppo_agent import PPOAgent

# --- adjust these to match your actual layer widths ---
LAYER_WIDTHS = [17, 30, 30, 30, 6]   # input=17 (obs), output=6 (actions)
NUM_EPISODES = 300
BATCH_SIZE   = 12                  # same default as ReinforceAgent

from src.agents.reinforce_agent import ReinforceAgent

agent = PPOAgent(
    layer_widths=LAYER_WIDTHS,
    batch_size=BATCH_SIZE,
    discount_factor=0.99,
)

env = gym.make("Walker2d-v5")

episode_times   = []   # wall time for entire episode
step_times_all  = []   # every individual forward-pass time
i=0
for ep in range(NUM_EPISODES):
    obs, _ = env.reset()
    ep_start = time.perf_counter()

    done = False
    while not done:
        t0 = time.perf_counter()
        action = agent.step(obs.tolist())
        step_times_all.append(time.perf_counter() - t0)

        obs, reward, terminated, truncated, _ = env.step(action)
        agent.reward(reward)
        done = terminated or truncated

    agent.episode_end()
    episode_times.append(time.perf_counter() - ep_start)
    i+=1
    if i%10==0:
        print(f"Episode {ep+1:3d}/{NUM_EPISODES}  |  "
              f"time: {episode_times[-1]:.3f}s  |  "
              f"steps: {len(step_times_all)}")

env.close()

# ── plots ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(10, 8))
fig.suptitle("Walker2d – ReinforceAgent timing diagnostic", fontsize=13)

axes[0].plot(episode_times, marker="o", linewidth=1, markersize=3)
axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Wall time (s)")
axes[0].set_title("Time per episode")
axes[0].grid(True, alpha=0.3)

axes[1].plot(step_times_all, linewidth=0.5, alpha=0.7)
axes[1].set_xlabel("Forward pass (global step index)")
axes[1].set_ylabel("Wall time (s)")
axes[1].set_title("Time per forward pass (agent.step)")
axes[1].grid(True, alpha=0.3)

# rolling average overlay so the trend is easy to see
window = max(1, len(step_times_all) // 100)
import numpy as np
rolled = np.convolve(step_times_all, np.ones(window)/window, mode="valid")
axes[1].plot(range(window-1, len(step_times_all)), rolled,
             color="red", linewidth=1.5, label=f"Rolling avg (w={window})")
axes[1].legend()

plt.tight_layout()
plt.savefig("timing_diagnostic.png", dpi=150)
plt.show()
print("Plot saved to timing_diagnostic.png")