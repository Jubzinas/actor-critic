import imageio
import numpy as np
import matplotlib.pyplot as plt

def show_video(eval_env, policy, filename="acrobot.gif", fps=30):
    images = []
    state, info = eval_env.reset()
    done = False

    while not done:
        frame = eval_env.render()
        images.append(frame)
        action, *_ = policy.act(state)   # ← changed
        state, reward, terminated, truncated, info = eval_env.step(action)
        done = terminated or truncated

    eval_env.close()
    imageio.mimwrite(filename, images, fps=fps, loop=0)
    return images


def evaluate_agent(env, max_steps, n_eval_episodes, policy):
    episode_rewards = []
    for episode in range(n_eval_episodes):
        state, info = env.reset()
        done = False
        total_rewards_ep = 0

        for step in range(max_steps):
            action, *_ = policy.act(state)   # ← changed
            new_state, reward, terminated, truncated, info = env.step(action)
            total_rewards_ep += reward

            if terminated or truncated:
                break
            state = new_state
        episode_rewards.append(total_rewards_ep)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    return mean_reward, std_reward

def plot_learning_and_scores_curves(scores, losses):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    ax1.plot(scores)
    ax1.set_title('Episode Rewards Over Time')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    
    ax2.plot(losses)
    ax2.set_title('Policy Loss Over Time')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    
    plt.tight_layout()
    plt.show()
    plt.savefig("learning_and_scores_curves.png")