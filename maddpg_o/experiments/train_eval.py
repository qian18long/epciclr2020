import argparse
import numpy as np
import tensorflow as tf
import time
import pickle
import imageio

import maddpg_local.common.tf_util as U
from maddpg_local.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple_sheep_wolf_bettershape_changefood", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=50, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=5, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=2, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg_local", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg_local", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default="", help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="./Jul_21/5_2_2_4_shapecorebound_normal23/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="./Jul_21/2_2_2_4_bettershape_changefood_200000_baseline/", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=True)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=10000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./simple_sheep_wolf/3_1_2_2/", help="directory where plot data is saved")
    return parser.parse_args()

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        # print(tf.shape(out))
        return out

def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, done_callback=scenario.done, info_callback=scenario.info)
    return env

def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(num_adversaries):
        trainers.append(trainer(
            "adv_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy=='ddpg')))
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            "agent_%d" % (i-num_adversaries), model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy=='ddpg')))
    return trainers


def train(arglist):
    with U.single_threaded_session():
        save_gifs = 1

        # Create environment
        curriculum = 0
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        # Create agent trainers

        if save_gifs:
            frames = []
            # frames.append(env.render('rgb_array')[0])

        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))





        # Initialize
        U.initialize()
        # U.load_state(arglist.load_dir) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.display or arglist.restore or arglist.benchmark:
            print('Loading previous state...')
            print(arglist.load_dir)
            U.load_state(arglist.load_dir)

        # U.load_state(arglist.load_dir)

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()
        obs_n = env.reset()
        episode_step = 0
        train_step = 0
        t_start = time.time()

        print('Starting iterations...')
        while True:
            # get action


            action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]
            # environment step
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            if (train_step % arglist.max_episode_len == 0):
                history_info = info_n
            else:
                iitem = 0
                for item in history_info:
                    history_info[iitem] += info_n[iitem]
                    iitem += 1

            if not train_step:
                info_all = history_info



            episode_step += 1
            done = all(done_n)

            # terminal = (episode_step >= arglist.max_episode_len)

            num = len(episode_rewards)//10000
            savedir = "./simple_sheep_wolf/5_3_2_4_50_curr"+str(num)+"/"
            max_len = 50+num*25

            if not curriculum:
                terminal = (episode_step >= arglist.max_episode_len)
            else:
                terminal = (episode_step >= max_len)


            # collect experience
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
            obs_n = new_obs_n

            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            if done or terminal:

                obs_n = env.reset()
                episode_step = 0
                episode_rewards.append(0)
                iitem = 0
                for item in history_info:
                    info_all[iitem] += history_info[iitem]
                    iitem += 1
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])

            # increment global step counter
            train_step += 1

            # for benchmarking learned policies

            if arglist.benchmark:
                for i, info in enumerate(info_n):
                    agent_info[-1][i].append(info_n['n'])
                if train_step > arglist.benchmark_iters and (done or terminal):
                    file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                    print('Finished benchmarking, now saving...')
                    with open(file_name, 'wb') as fp:
                        pickle.dump(agent_info[:-1], fp)
                    break
                continue

            # for displaying learned policies
            # print(done_n)
            # print(info_n)
            '''
            if num_adversaries == 0:
                print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                                                                                          train_step, len(episode_rewards), np.mean(episode_rewards[-2:]), round(time.time()-t_start, 3)))
            else:
                print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                                                                                                                    train_step, len(episode_rewards), np.mean(episode_rewards[-2:]),
                                                                                                                    [np.mean(rew[-2:]) for rew in agent_rewards], round(time.time()-t_start, 3)))

            '''

            if arglist.display:
                if save_gifs:
                    frames.append(env.render('rgb_array')[0])
                time.sleep(0.1)
                env.render()
                if terminal or done:
                    print (history_info)
                    if num_adversaries == 0:
                        print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                            train_step, len(episode_rewards), np.mean(episode_rewards[-2:]), round(time.time()-t_start, 3)))
                    else:
                        print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                            train_step, len(episode_rewards), np.mean(episode_rewards[-2:]),
                            [np.mean(rew[-2:]) for rew in agent_rewards], round(time.time()-t_start, 3)))
                if len(episode_rewards) > arglist.num_episodes:
                    break

                continue
            '''
            if terminal and (len(episode_rewards) % arglist.save_rate == 0) and len(episode_rewards)>1000:
                time.sleep(0.1)
                env.render()
                continue
            '''

            # update all trainers, if not in display or benchmark mode
            loss = None
            for agent in trainers:
                agent.preupdate()
            for agent in trainers:
                loss = agent.update(trainers, train_step) #all agent update


            # save model, display training output

            if terminal and (len(episode_rewards) % arglist.save_rate == 0):
                if not curriculum:
                    U.save_state(arglist.save_dir, saver=saver)
                else:
                    U.save_state(savedir, saver=saver)
                print (info_all)
                info_all = info_n
                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]), round(time.time()-t_start, 3)))
                else:
                    print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                        [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time()-t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

            # saves final episode reward for plotting training curve later

            if len(episode_rewards) > arglist.num_episodes or train_step>3000000:
                print('end')

                rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)

                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                break
        if save_gifs:
            gif_num = 0
            imageio.mimsave(arglist.load_dir+'5_3_2_4normal_noshape.gif',
                            frames, duration=1/5)

if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
