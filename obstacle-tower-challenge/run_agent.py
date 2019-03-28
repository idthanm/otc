from obstacle_tower_env import ObstacleTowerEnv
import sys
import numpy as np
import argparse
import dopamine.discrete_domains.create_otc_env as create_environment
from dopamine.agents.dqn import dqn_agent
from dopamine.agents.implicit_quantile import implicit_quantile_agent
from dopamine.agents.rainbow import rainbow_agent
from dopamine.discrete_domains import checkpointer
import tensorflow as tf


def initialize_checkpointer(checkpoint_dir, checkpoint_file_prefix, agent):
    """Reloads the latest checkpoint if it exists.

    This method will first create a `Checkpointer` object and then call
    `checkpointer.get_latest_checkpoint_number` to determine if there is a valid
    checkpoint in self._checkpoint_dir, and what the largest file number is.
    If a valid checkpoint file is found, it will load the bundled data from this
    file and will pass it to the agent for it to reload its data.
    If the agent is able to successfully unbundle, this method will verify that
    the unbundled data contains the keys,'logs' and 'current_iteration'. It will
    then load the `Logger`'s data from the bundle, and will return the iteration
    number keyed by 'current_iteration' as one of the return values (along with
    the `Checkpointer` object).

    Args:
      checkpoint_file_prefix: str, the checkpoint file prefix.

    Returns:
      start_iteration: int, the iteration number to start the experiment from.
      experiment_checkpointer: `Checkpointer` object for the experiment.
    """
    checkpointer_ = checkpointer.Checkpointer(checkpoint_dir,
                                                   checkpoint_file_prefix)
    start_iteration = 0
    # Check if checkpoint exists. Note that the existence of checkpoint 0 means
    # that we have finished iteration 0 (so we will start from iteration 1).
    latest_checkpoint_version = checkpointer.get_latest_checkpoint_number(
        checkpoint_dir)
    if latest_checkpoint_version >= 0:
        experiment_data = checkpointer_.load_checkpoint(
            latest_checkpoint_version)
        agent.unbundle(checkpoint_dir, latest_checkpoint_version, experiment_data)

def create_agent(sess, environment, agent_name=None, summary_writer=None,
                 debug_mode=False):
  """Creates an agent.

  Args:
    sess: A `tf.Session` object for running associated ops.
    environment: A gym environment (e.g. Atari 2600).
    agent_name: str, name of the agent to create.
    summary_writer: A Tensorflow summary writer to pass to the agent
      for in-agent training statistics in Tensorboard.
    debug_mode: bool, whether to output Tensorboard summaries. If set to true,
      the agent will output in-episode statistics to Tensorboard. Disabled by
      default as this results in slower training.

  Returns:
    agent: An RL agent.

  Raises:
    ValueError: If `agent_name` is not in supported list.
  """
  assert agent_name is not None
  if not debug_mode:
    summary_writer = None
  if agent_name == 'dqn':
    return dqn_agent.DQNAgent(sess, num_actions=environment.action_space.n,
                              summary_writer=summary_writer)
  elif agent_name == 'rainbow':
    return rainbow_agent.RainbowAgent(
        sess, num_actions=environment.action_space.n,
        summary_writer=summary_writer)
  elif agent_name == 'implicit_quantile':
    return implicit_quantile_agent.ImplicitQuantileAgent(
        sess, num_actions=environment.action_space.n,
        summary_writer=summary_writer)
  else:
    raise ValueError('Unknown agent: {}'.format(agent_name))


def run_episode(initial_observation, env, agent):
    is_terminal = False
    episode_reward = 0.0
    action = agent.begin_episode(initial_observation)
    
    while not is_terminal:
        observation, reward, is_terminal, _ = env.step(action)
        action = agent.step(reward, observation)
        episode_reward += reward
        
    return episode_reward


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('environment_filename', default='../ObstacleTower/obstacletower', nargs='?')
    parser.add_argument('--docker_training', action='store_true')
    parser.set_defaults(docker_training=False)
    args = parser.parse_args()

    env = create_environment.create_otc_environment(environment_filename=args.environment_filename,
                                                    docker_training=args.docker_training, realtime_mode=True)
    sess = tf.Session('', config=tf.ConfigProto(allow_soft_placement=True))
    agent = create_agent(sess=sess, environment=env, agent_name='rainbow')
    initialize_checkpointer("..\\checkpoints", 'ckpt', agent)


    while True:
        initial_observation = env.reset()
        episode_reward = run_episode(initial_observation, env, agent)
        print("Episode reward: " + str(episode_reward))

