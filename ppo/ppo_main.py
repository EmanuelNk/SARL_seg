# import gym
from tokenize import PlainToken
import numpy as np
import random
from seg_env import preprocess_data
from ppo_torch import Agent, transform_obs
from utils import plot_learning_curve
import matplotlib.pyplot as plt
import json

envs = preprocess_data('coffee', display=False)

if __name__ == '__main__':

    def save_episode_data_json(episode, durations):
        data = {"episode": episode, "durations": durations}
        filename = "ppo/episode_info/ppo_episode_data_dict.json"
        with open(filename, 'w') as f:
            json.dump(data, f)

    def load_episode_data_json():
        filename = "ppo/episode_info/ppo_episode_data_dict.json"
        with open(filename, 'r') as f:
            data = json.load(f)
        return data
    

    env = random.choice(envs)
    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    agent = Agent(n_actions=4, batch_size=batch_size, 
                    alpha=alpha, n_epochs=n_epochs, 
                    input_dims=[3,112,112])

    # print the device
    print(f"### device: {agent.actor.device} ###") 

    n_games = 300

    figure_file = 'plots/cartpole.png'

    best_score = env.best_reward
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0
    episode_durations = []
    episode_data = load_episode_data_json()
    episode = episode_data['episode'] + 1
    episode_durations = episode_data['durations']

    for i in range(episode, n_games):
        observation = env.reset()
        done = False
        score = 0
        t = 0
        while not done:
            t += 1
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done = env.step(action)
            if reward > env.best_reward:
                env.best_reward = reward
            n_steps += 1
            score += reward
            agent.remember(transform_obs(observation), action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_

            if t%100 == 0:
                print(f"Episode {i} | Step {t} | Reward {reward} | Best Reward {env.best_reward}")
        
        episode_durations.append(t)

        if i%2 == 0:
            agent.save_models()
            save_episode_data_json(i, episode_durations)
            
        plt.imshow(observation)
        plt.title(f"Episode {i} | Step {t} Done!")
        plt.show()
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps, 'learning_steps', learn_iters)
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)

