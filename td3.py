import mujoco_py
from copy import deepcopy
import time

import gym
import gym.wrappers
import numpy as np

import jittor as jt
from jittor import nn
jt.flags.use_cuda = 1


class FLAGS:
    class TD3:
        gamma = 0.99
        lr = 3e-4
        policy_noise = 0.2
        noise_clip = 0.5
        policy_freq = 2
        batch_size = 256

        tau = 0.005

    env_id = 'HalfCheetah-v2'
    n_test_episodes = 10
    max_buf_size = 1000_000
    n_iters = 1_000_000
    n_expl_steps = 25_000
    expl_noise = 0.1


def polyak_copy(origin, target, tau):
    for origin_param, target_param in zip(origin.parameters(), target.parameters()):
        target_param.assign((target_param * (1 - tau) + origin_param * tau).stop_grad())


class TD3Trainer:
    def __init__(self, policy, policy_target, qfns, qfns_target, sampler, *, FLAGS):
        super().__init__()
        self.FLAGS = FLAGS
        self.policy = policy
        self.policy_target = policy_target
        self.qfns = qfns
        self.qfns_target = qfns_target
        self.sampler = sampler

        self.qfns_opt = nn.Adam(sum([qfn.parameters() for qfn in self.qfns], []), FLAGS.lr)
        self.policy_opt = nn.Adam(self.policy.parameters(), FLAGS.lr)

        self.n_batches = 0

    def training_step(self, batch, batch_idx):
        noises = jt.array(np.random.randn(*batch['action'].shape).astype(np.float32)) * self.FLAGS.policy_noise  # TODO: use jt randomness
        noises = noises.clamp(-self.FLAGS.noise_clip, self.FLAGS.noise_clip)
        next_actions = self.policy_target(batch['next_observation']).add(noises).clamp(-1, 1)
        next_qfs = [qfn(batch['next_observation'], next_actions) for qfn in self.qfns_target]
        min_next_qf = jt.minimum(next_qfs[0], next_qfs[1])
        qf_ = (batch['reward'] + (1 - batch['done'].float32()) * self.FLAGS.gamma * min_next_qf).detach()

        qfn_losses = [nn.mse_loss(qfn(batch['observation'], batch['action']), qf_) for qfn in self.qfns]

        self.qfns_opt.step(qfn_losses[0] + qfn_losses[1])

        if self.n_batches % self.FLAGS.policy_freq == 0:
            policy_loss = -self.qfns[0](batch['observation'], self.policy(batch['observation'])).mean()
            self.policy_opt.step(policy_loss)

            polyak_copy(self.policy, self.policy_target, self.FLAGS.tau)
            for qfn, qfn_target in zip(self.qfns, self.qfns_target):
                polyak_copy(qfn, qfn_target, self.FLAGS.tau)
        return {'loss': [qfn_loss.data for qfn_loss in qfn_losses]}

    def step(self):
        self.n_batches += 1

        batch = self.sampler(self.FLAGS.batch_size)
        output = self.training_step(batch, self.n_batches)

        if self.n_batches % 1000 == 0:
            print(time.time(), self.n_batches, output)


def evaluate(t, policy, env):
    returns = []

    for _ in range(FLAGS.n_test_episodes):
        observation, done, return_ = env.reset(), False, 0.
        while not done:
            action = policy.get_actions(observation)
            next_observation, reward, done, _ = env.step(action)
            observation = next_observation
            return_ += reward

        returns.append(return_)
    mean = np.mean(returns)
    print(f"Evaluation at step {t} over {FLAGS.n_test_episodes} episodes: {mean:.0f}")


class MLPPolicy(nn.Module):
    def __init__(self, dim_observation, dim_action):
        super().__init__()
        self.fc1 = nn.Linear(dim_observation, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, dim_action)

    def execute(self, x):
        x = self.fc1(x)
        x = nn.relu(x)
        x = self.fc2(x)
        x = nn.relu(x)
        x = self.fc3(x)
        return jt.tanh(x)

    def get_actions(self, observations):
        observations = jt.float32(observations)
        if len(observations.shape) == 1:
            observations = observations.reshape(1, -1)
            return self(observations).data[0]
        return self(observations).data


class MLPQFunction(nn.Module):
    def __init__(self, dim_observation, dim_action):
        super().__init__()
        self.fc1 = nn.Linear(dim_observation + dim_action, 256)
        # self.fc11 = nn.Linear(dim_observation, 256)
        # self.fc12 = nn.Linear(dim_action, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def execute(self, observations, actions):
        x = self.fc1(jt.contrib.concat([observations, actions], dim=-1))
        x = nn.relu(x)
        x = self.fc2(x)
        x = nn.relu(x)
        x = self.fc3(x)
        return x.squeeze(1)


class ReplayBuffer:
    def __init__(self, env, max_buf_size):
        shapes = {
            'observation': (jt.float32, env.observation_space.shape),
            'action': (jt.float32, env.action_space.shape),
            'next_observation': (jt.float32, env.observation_space.shape),
            'reward': (jt.float32, ()),
            'done': (jt.bool, ()),
        }
        self.data = {k: jt.zeros([max_buf_size, *shape], dtype=dtype) for k, (dtype, shape) in shapes.items()}
        self.len = 0
        self.max_buf_size = max_buf_size

    def __len__(self):
        return self.len

    def sample(self, n_samples=1):
        indices = np.random.randint(len(self), size=(n_samples,), dtype=np.int64)
        return {k: v[indices] for k, v in self.data.items()}

    def add_transition(self, transition):
        for key in self.data.keys():
            # TODO: performance of a[b]=c
            # TODO: a[b] = np.float64(1.)
            self.data[key].data[self.len % self.max_buf_size] = np.array(transition[key])
        self.len += 1


def main():
    make_env = lambda: gym.wrappers.RescaleAction(gym.make(FLAGS.env_id), -1, 1)
    env = make_env()

    dim_observation = env.observation_space.shape[0]
    dim_action = env.action_space.shape[0]

    buffer = ReplayBuffer(env, FLAGS.max_buf_size)
    policy = MLPPolicy(dim_observation, dim_action)
    qfns = [
        MLPQFunction(dim_observation, dim_action),
        MLPQFunction(dim_observation, dim_action),
    ]

    policy_target = MLPPolicy(dim_observation, dim_action)
    qfns_target = [
        MLPQFunction(dim_observation, dim_action),
        MLPQFunction(dim_observation, dim_action),
    ]
    polyak_copy(policy, policy_target, 1)
    for qfn, qfn_target in zip(qfns, qfns_target):
        polyak_copy(qfn, qfn_target, 1)

    algo_policy = TD3Trainer(policy, policy_target, qfns, qfns_target, FLAGS=FLAGS.TD3, sampler=buffer.sample)

    print("start training")
    observation = env.reset()
    for t in range(FLAGS.n_iters):
        if t % 10_000 == 0:
            evaluate(t, policy, make_env())

        if t < FLAGS.n_expl_steps:
            action = env.action_space.sample()
        else:
            action = policy.get_actions(observation) + FLAGS.expl_noise * np.random.randn(dim_action)
            action = np.clip(action, -1, 1)
        next_observation, reward, done, info = env.step(action)

        real_done = done and not info['TimeLimit.truncated']
        buffer.add_transition({'observation': observation, 'action': action, 'next_observation': next_observation,
                               'reward': reward, 'done': real_done})
        if t >= FLAGS.n_expl_steps:
            algo_policy.step()

        observation = env.reset() if done else next_observation
        jt.sync_all()


if __name__ == '__main__':
    main()
