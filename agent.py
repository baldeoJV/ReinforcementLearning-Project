from buffer import ReplayBuffer
from model import Model, soft_update
import torch
import torch.optim as optim
import torch.nn.functional as F
import datetime
import time
from torch.utils.tensorboard import SummaryWriter
import random
import os
import cv2
import numpy as np

class Agent():

    def __init__(self, env, hidden_layer, learning_rate, step_repeat, gamma):
        
        self.env = env

        self.step_repeat = step_repeat

        self.gamma = gamma

        obs, info = self.env.reset()

        obs = self.process_observation(obs)

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        print(f'Loaded model on device {self.device}')

        self.memory = ReplayBuffer(max_size=500000, input_shape=obs.shape, device=self.device)

        self.model = Model(action_dim=env.action_space.n, hidden_dim=hidden_layer, observation_shape=obs.shape).to(self.device)
        
        self.target_model = Model(action_dim=env.action_space.n, hidden_dim=hidden_layer, observation_shape=obs.shape).to(self.device)

        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.learning_rate = learning_rate


    def process_observation(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1)
        return obs


    def test(self):

        self.model.load_the_model()

        obs, info = self.env.reset()

        done = False
        obs, info = self.env.reset()
        obs = self.process_observation(obs)

        episode_reward = 0

        while not done:

            if random.random() < 0.05:
                action = self.env.action_space.sample()
            else:
                q_values = self.model.forward(obs.unsqueeze(0).to(self.device))[0]
                action = torch.argmax(q_values, dim=-1).item()
            
            reward = 0

            for i in range(self.step_repeat):
                reward_temp = 0

                next_obs, reward_temp, done, truncated, info = self.env.step(action=action)

                reward += reward_temp

                # frame = self.env.env.env.render() 

                # resized_frame = cv2.resize(frame, (500, 400))

                # resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR)

                # cv2.imshow("Pong AI", resized_frame)

                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break

                # # time.sleep(0.05) #adjust speed to control playback speed

                if(done):
                    break
            
            obs = self.process_observation(next_obs)

            episode_reward += reward
    

    def train(self, episodes, max_episode_steps, summary_writer_suffix, batch_size, epsilon, epsilon_decay, min_epsilon):

            # Save TensorBoard logs in the same directory as the script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            summary_writer_name = os.path.join(
                script_dir, 'runs',
                f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{summary_writer_suffix}'
            )
            writer = SummaryWriter(summary_writer_name)

            if not os.path.exists('models'):
                os.makedirs('models')

            total_steps = 0
            reward_history, loss_history, qvalue_history = [], [], []

            print(f"TensorBoard logs → {summary_writer_name}")

            for episode in range(episodes):
                done = False
                episode_reward = 0
                obs, info = self.env.reset()
                obs = self.process_observation(obs)
                episode_steps = 0
                episode_losses, episode_qvalues, td_errors, action_counts = [], [], [], []

                episode_start_time = time.time()

                while not done and episode_steps < max_episode_steps:
                    # ε-greedy action selection
                    if random.random() < epsilon:
                        action = self.env.action_space.sample()
                    else:
                        q_values = self.model.forward(obs.unsqueeze(0).to(self.device))[0]
                        action = torch.argmax(q_values, dim=-1).item()
                        episode_qvalues.append(q_values.mean().item())

                    action_counts.append(action)

                    reward = 0
                    for _ in range(self.step_repeat):
                        next_obs, reward_temp, done, truncated, info = self.env.step(action=action)
                        reward += reward_temp
                        if done:
                            break

                    next_obs = self.process_observation(next_obs)
                    self.memory.store_transition(obs, action, reward, next_obs, done)
                    obs = next_obs

                    episode_reward += reward
                    episode_steps += 1
                    total_steps += 1

                    # Training updates
                    if self.memory.can_sample(batch_size):
                        observations, actions, rewards, next_observations, dones = self.memory.sample_buffer(batch_size)
                        dones = dones.unsqueeze(1).float()

                        # Q(s,a)
                        q_values = self.model(observations)
                        actions = actions.unsqueeze(1).long()
                        qsa_batch = q_values.gather(1, actions)

                        # Q-target(s’, a’)
                        next_actions = torch.argmax(self.model(next_observations), dim=1, keepdim=True)
                        next_q_values = self.target_model(next_observations).gather(1, next_actions)
                        target_b = rewards.unsqueeze(1) + (1 - dones) * self.gamma * next_q_values

                        # TD error & loss
                        td_error = (target_b.detach() - qsa_batch)
                        loss = F.mse_loss(qsa_batch, target_b.detach())

                        episode_losses.append(loss.item())
                        td_errors.extend(td_error.abs().detach().cpu().numpy().flatten().tolist())


                        # Optimization step
                        self.model.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                        if episode_steps % 4 == 0:
                            soft_update(self.target_model, self.model)

                        # Log per-step metrics
                        writer.add_scalar("Loss/step", loss.item(), total_steps)
                        writer.add_scalar("QValue/mean", q_values.mean().item(), total_steps)
                        writer.add_scalar("QValue/max", q_values.max().item(), total_steps)
                        writer.add_scalar("TD_Error/mean", td_error.abs().mean().item(), total_steps)

                # Model checkpoint
                self.model.save_the_model()

                # Episode metrics
                episode_time = time.time() - episode_start_time
                avg_loss = np.mean(episode_losses) if episode_losses else 0
                avg_qvalue = np.mean(episode_qvalues) if episode_qvalues else 0
                avg_td_error = np.mean(td_errors) if td_errors else 0
                fps = episode_steps / episode_time if episode_time > 0 else 0
                avg_reward = np.mean(reward_history[-10:]) if reward_history else 0

                reward_history.append(episode_reward)
                loss_history.append(avg_loss)
                qvalue_history.append(avg_qvalue)

                # TensorBoard logs (episode level)
                writer.add_scalar('Score/Episode', episode_reward, episode)
                writer.add_scalar('Score/Avg10', avg_reward, episode)
                writer.add_scalar('Loss/Episode', avg_loss, episode)
                writer.add_scalar('QValue/Avg', avg_qvalue, episode)
                writer.add_scalar('TD_Error/EpisodeMean', avg_td_error, episode)
                writer.add_scalar('Epsilon', epsilon, episode)
                writer.add_scalar('Performance/FPS', fps, episode)
                writer.add_scalar('Performance/EpisodeTime', episode_time, episode)

                # Action distribution histogram
                writer.add_histogram('Action/Distribution', np.array(action_counts), episode)

                # Console output
                print(f"\nEpisode {episode + 1}/{episodes}")
                print(f"  Reward: {episode_reward:.2f}  |  Avg(10): {avg_reward:.2f}")
                print(f"  Avg Loss: {avg_loss:.6f}  |  Avg Q: {avg_qvalue:.3f}  |  TD Error: {avg_td_error:.4f}")
                print(f"  Steps: {episode_steps}  |  FPS: {fps:.2f}  |  Time: {episode_time:.2f}s")
                print(f"  Epsilon: {epsilon:.4f}")

                # Epsilon decay
                if epsilon > min_epsilon:
                    epsilon *= epsilon_decay

            writer.close()
            print("\n✅ Training complete! You can now view metrics with:")
            print(f"   tensorboard --logdir {os.path.join(script_dir, 'runs')}")
