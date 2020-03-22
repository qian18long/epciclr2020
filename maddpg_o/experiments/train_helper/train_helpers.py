# sys.path = ['..'] + sys.path
# sys.path = ['../../mpe_local'] + sys.path
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.contrib import layers
import math
from maddpg_o.maddpg_local.micro.maddpg import MADDPGAgentMicroSharedTrainer
import maddpg_o.maddpg_local.common.tf_util as U
from .model_v3_test3 import mlp_model_agent_p, mlp_model_adv_p, mlp_model_agent_q, mlp_model_adv_q, mlp_model, mean_field_adv_q_model, mean_field_agent_q_model
from .model_v3_numbered import mlp_model_agent_p_numbered, mlp_model_adv_p_numbered, mlp_model_agent_q_numbered, mlp_model_adv_q_numbered
import argparse
import time
import re
# import ray
import multiprocessing
# import pathos.multiprocessing as mp
# from multiprocessing import Pool
# sys.path.append('..')
# from maddpg_local.common import tf_util as U
from functools import partial
from maddpg_o.experiments.train_helper.union_replay_buffer import UnionReplayBuffer
import numpy as np
import imageio
import queue
import joblib
import tempfile
import random
import gc


FLAGS = None

# import multiagent
# print(multiagent.__file__)
# print(Scenario)

# ray.init()

# load model
# N_GOOD, N_ADV, N_LAND should align with the environment
N_GOOD = None
N_ADV = None
# N_LAND = num_landmarks+num_food+num_forests
N_LANDMARKS = None
N_FOOD = None
N_FORESTS = None
N_LAND = None

ID_MAPPING = None

INIT_WEIGHTS = None
GOOD_SHARE_WEIGHTS = False
ADV_SHARE_WEIGHTS = False
SHARE_WEIGHTS = None
CACHED_WEIGHTS = {}

WEIGHT_STACK = False

GRAPHS = []
SESSIONS = []
TRAINERS = []

CLUSTER = None
SERVERS = None

PERTURBATION = None
NEW_IDS = []


def exp_to_bytes(f, all_data):
    flat_data = []
    for data_n in all_data:
        if type(data_n) == list:
            for data in data_n:
                flat_data.append(data.flatten())
        else:
            flat_data.append(data_n.flatten())
    total_length = 0
    # print(len(flat_data))

    flat_data = np.concatenate(flat_data).astype(np.float32)
    b = flat_data.tobytes()
    # f.write(b)
    total_length = len(b)
    # s = 1
    # while s < total_length:
    #     s *= 2
    # b += b'f' * (s - total_length)
    # total_length = s
    # assert len(b) == s
    f.write(b)
    # print("tt", total_numbers)
    # f.flush()
    return total_length, flat_data


def bytes_to_exp(f, n):
    flat_data = np.fromfile(f, dtype=np.float32, count=n)
    # all_data = []
    # k = 0
    # for i in range(5):
    #     data_n = []
    #     for j in range(n):
    #         data_n.append(flat_data[k])
    #         k += 1
    #     all_data.append(data_n)
    return flat_data


def format_time(ti):
    h, m = divmod(ti, 3600)
    m, s = divmod(m, 60)
    s, ms = divmod(s, 1)
    return "{:2d}h{:2d}m{:2d}s.{:3d}".format(int(h), int(m), int(s), int(ms * 1000))


def register_environment(n_good, n_adv, n_landmarks, n_food, n_forests, init_weights, id_mapping=None):
    global N_GOOD, N_ADV, N_LAND, N_LANDMARKS, N_FOOD, N_FORESTS, ID_MAPPING, INIT_WEIGHTS
    N_GOOD = n_good
    N_ADV = n_adv
    N_LANDMARKS = n_landmarks
    N_FOOD = n_food
    N_FORESTS = n_forests
    N_LAND = N_LANDMARKS + N_FOOD + N_FORESTS
    INIT_WEIGHTS = init_weights
    ID_MAPPING = id_mapping
    # print("SHARE_WEIGHTS", SHARE_WEIGHTS)


def name_encode(name, convert):
    # print(name)

    def agent_decode(name):
        # if name == "self":
        #     return last
        match = re.match(r'agent_(\d+)', name)
        if match:
            return int(match.group(1))
        match = re.match(r'good(\d+)', name)
        if match:
            return int(match.group(1)) + N_ADV
        match = re.match(r'adv(\d+)', name)
        if match:
            return int(match.group(1))
        return name

    names = name.split('/')
    ret = []
    for name in names:
        decoded = agent_decode(name)
        # if type(decoded) == int:
        #     # decoded = id_reverse_mapping[decoded] if convert else decoded
        #     last = decoded
        ret.append(decoded)

    last = None

    is_new = None

    for i in range(len(ret)):
        if type(ret[i]) == int:
            if is_new is None:
                is_new = ret[i] in NEW_IDS
            if convert:
                ret[i] = ID_MAPPING[ret[i]]
            if last == ret[i]:
            # if last == ret[i]:
                return None, None
            else:
                last = ret[i]
            ret[i] = str(ret[i])

    # print('/'.join(ret))
    return '/'.join(ret), is_new


def add_perturbation(weight):
    if PERTURBATION is None:
        return weight
    std = np.std(weight)
    return weight + np.random.normal(0., PERTURBATION * std, weight.shape)


def make_env(scenario_name, arglist, benchmark=False):
    import importlib
    from mpe_local.multiagent.environment import MultiAgentEnv

    module_name = "mpe_local.multiagent.scenarios.{}".format(scenario_name)
    scenario_class = importlib.import_module(module_name).Scenario
    # load scenario from script
    # print(Scenario.__module__.__file__)
    ratio = 1.0 if FLAGS.map_size == "normal" else 2.0
    scenario = scenario_class(n_good=N_GOOD, n_adv=N_ADV, n_landmarks=N_LANDMARKS, n_food=N_FOOD, n_forests=N_FORESTS,
                              no_wheel=FLAGS.no_wheel, sight=FLAGS.sight, alpha=FLAGS.alpha, ratio=ratio)
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                            scenario.observation, scenario.benchmark_data, export_episode=FLAGS.save_gif_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                            scenario.observation, done_callback=scenario.done, info_callback=scenario.info,
                            export_episode=FLAGS.save_gif_data)
    return env


def make_session(graph, num_cpu):
    # print("num_cpu:", num_cpu)
    tf_config = tf.ConfigProto(
        # device_count={"CPU": num_cpu},
        inter_op_parallelism_threads=num_cpu,
        intra_op_parallelism_threads=num_cpu,
        log_device_placement=False)
    tf_config.gpu_options.allow_growth = True
    return tf.Session(graph=graph, config=tf_config)
    # return tf.Session(target=server.target, graph=graph, config=tf_config)


def get_trainer(side, i, scope, env, obs_shape_n):
    trainer = MADDPGAgentMicroSharedTrainer
    policy = FLAGS.adv_policy if side == "adv" else FLAGS.good_policy
    share_weights = FLAGS.adv_share_weights if side == "adv" else FLAGS.good_share_weights
    if policy == "att-maddpg":
        model_p = partial(mlp_model_adv_p if side == "adv" else mlp_model_agent_p, n_good=N_GOOD, n_adv=N_ADV,
                          n_land=N_LAND, index=i, share_weights=share_weights)
        model_q = partial(mlp_model_adv_q if side == "adv" else mlp_model_agent_q, n_good=N_GOOD, n_adv=N_ADV,
                          n_land=N_LAND, index=i, share_weights=share_weights)
    elif policy == "PC":
        model_p = partial(mlp_model_adv_p_numbered if side == "adv" else mlp_model_agent_p_numbered, n_good=N_GOOD, n_adv=N_ADV,
                          n_land=N_LAND, index=i, share_weights=share_weights)
        model_q = partial(mlp_model_adv_q_numbered if side == "adv" else mlp_model_agent_q_numbered, n_good=N_GOOD, n_adv=N_ADV,
                          n_land=N_LAND, index=i, share_weights=share_weights)
    elif policy == "maddpg":
        model_p = mlp_model
        model_q = mlp_model
    elif policy == "mean_field":
        model_p = mlp_model
        model_q = partial(mean_field_adv_q_model if side == "adv" else mean_field_agent_q_model, n_good=N_GOOD,
                          n_adv=N_ADV, n_land=N_LAND, index=i)
    else:
        raise NotImplementedError
    # print(obs_shape_n)
    num_units = (FLAGS.adv_num_units if side == "adv" else FLAGS.good_num_units) or FLAGS.num_units
    return trainer(scope, model_p, model_q, obs_shape_n, env.action_space, i, FLAGS, num_units, local_q_func=False)


def get_adv_trainer(i, scope, env, obs_shape_n):
    return get_trainer("adv", i, scope, env, obs_shape_n)


def get_good_trainer(i, scope, env, obs_shape_n):
    return get_trainer("good", i, scope, env, obs_shape_n)


def show_size():
    s = 0
    for var in tf.trainable_variables():
        shape = var.get_shape()
        tot = 1
        for dim in shape:
            tot *= dim
        # if tot > 5000:
        #     print(tot, shape, var.name)
        s += tot
    # print("total size:", s)


def touch_path(path):
    dirname = os.path.dirname(path)
    if any(dirname):
        os.makedirs(dirname, exist_ok=True)


def load_weights(load_path):
    import joblib
    global CACHED_WEIGHTS

    CACHED_WEIGHTS.update(joblib.load(load_path))


def clean(d):
    rd = {}
    for k, v in d.items():
        # if v.shape[0] == 456 or v.shape[0] == 1552:
        #     print(k, v.shape)
        if type(k) == tuple:
            rd[k[0]] = v
        else:
            rd[k] = v
    return rd


def load_all_weights(load_dir, n):
    global CACHED_WEIGHTS
    CACHED_WEIGHTS = {}
    for i in range(n):
        # print(os.path.join(load_dir[i], "agent{}.trainable-weights".format(i)))
        load_weights(os.path.join(load_dir[i], "agent{}.trainable-weights".format(i)))
    # print(CACHED_WEIGHTS)
    CACHED_WEIGHTS = clean(CACHED_WEIGHTS)
    # for weight, value in CACHED_WEIGHTS.items():
    #     if weight[-7:] == "gamma:0" or weight[-6:] == "beta:0":
    #         print(weight, value)
    # print(CACHED_WEIGHTS.keys())


def parse_args(add_extra_flags=None):
    parser = argparse.ArgumentParser(
        "Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str,
                        default="grassland",
                        help="name of the scenario script")
    parser.add_argument("--map-size", type=str, default="normal")
    parser.add_argument("--sight", type=float, default=100)
    parser.add_argument("--no-wheel", action="store_true", default=False)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--show-attention", action="store_true", default=False)
    parser.add_argument("--max-episode-len", type=int,
                        default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int,
                        default=200000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int,
                        default=2, help="number of adversaries")
    parser.add_argument("--num-good", type=int,
                        default=2, help="number of good")
    parser.add_argument("--num-food", type=int,
                        default=4, help="number of food")
    parser.add_argument("--good-policy", type=str,
                        default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str,
                        default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float,
                        default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64,
                        help="number of units in the mlp")
    parser.add_argument("--good-num-units", type=int)
    parser.add_argument("--adv-num-units", type=int)
    parser.add_argument("--n-cpu-per-agent", type=int, default=1)
    parser.add_argument("--good-share-weights", action="store_true", default=False)
    parser.add_argument("--adv-share-weights", action="store_true", default=False)
    parser.add_argument("--use-gpu", action="store_true", default=False)
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="./test/",
                        help="directory in which training state and model should be saved")
    parser.add_argument("--train-rate", type=int, default=100,
                        help="save model once every time this many episodes are completed")
    parser.add_argument("--save-rate", type=int, default=1000,
                        help="save model once every time this many episodes are completed")
    parser.add_argument("--checkpoint-rate", type=int, default=0)
    parser.add_argument("--load-dir", type=str, default="./test/",
                        help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--save-gif-data", action="store_true", default=False)
    parser.add_argument("--render-gif", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=10000,
                        help="number of iterations run for benchmarking")

    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--save-summary", action="store_true", default=False)
    parser.add_argument("--timeout", type=float, default=0.02)

    if add_extra_flags is not None:
        parser = add_extra_flags(parser)

    return parser.parse_args()


def calc_size(var_list):
    s = 0
    for var in var_list:
        shape = var.get_shape()
        tot = 1
        for dim in shape:
            tot *= dim
        # if tot > 5000:
        #     print(tot, shape, var.name)
        s += tot
    return s

# @ray.remote
class Agent(multiprocessing.Process):
    def __init__(self, index, n, obs_batch_size, obs_shape, update_event, save_event, save_dir, load_dir,
                 cached_weights, num_cpu, get_trainer, main_conn, obs_queue, env_conns, use_gpu, timeout,
                 attention_mode, agent_sends, agent_recvs, tmp_file, obs_len, act_len, batch_size, train):
        multiprocessing.Process.__init__(self, daemon=True)
        # self.sess = SESSIONS[i]
        # self.graph = GRAPHS[i]
        self.index = index
        self.n = n
        self.obs_batch_size = obs_batch_size
        self.obs_shape = obs_shape
        self.scope = "agent_runner_{}".format(index)
        self.num_cpu = num_cpu
        self.get_trainer = get_trainer
        self.main_conn = main_conn
        self.obs_queue = obs_queue
        self.env_conns = env_conns
        self.save_dir = save_dir
        self.load_dir = load_dir
        self.trainer = None
        self.sess = None
        self.graph = None
        self.var_list = None
        self.cached_weights = cached_weights
        self.use_gpu = use_gpu
        self.update_event = update_event
        self.save_event = save_event
        self.sum_batch_size = None
        self.tot_batch = None
        self.timeout = timeout
        self.attention_mode = attention_mode
        self.agent_sends = agent_sends
        self.agent_recvs = agent_recvs
        self.tmp_file = open(tmp_file, "rb")
        self.obs_len = obs_len
        self.act_len = act_len
        self.batch_size = batch_size
        self.train = train
        # self.replay_buffer = ReplayBuffer(int(1e6))

    # def get_trainer(self):
    #     return self.trainer

    def build(self):
        # U.set_session(self.sess)

        # with self.graph.as_default():
        #     with self.sess:
        self.trainer = self.get_trainer()
        # print(self.trainer())
        self.var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.trainable_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        # print(self.index, len(self.trainable_var_list), calc_size(self.trainable_var_list))
        # for var in self.trainable_var_list:
        #     print(self.graph, var.graph, var.name)

        return None

    def restore_weights(self, load_dir):
        # print(self.index, len(self.var))
        U.load_variables(os.path.join(load_dir, "agent{}.weights".format(self.index)),
                         variables=self.var_list,
                         sess=self.sess)
        return None

    def restore_trainable_weights(self, weights):
        restores = []
        for v in self.trainable_var_list:
            name, is_new = name_encode(v.name, convert=True)
            if name is None:
                continue
            w = weights[name]
            if is_new:
                w = add_perturbation(w)
            restores.append(v.assign(w))
        self.sess.run(restores)

    def action(self, obs):
        return self.trainer.batch_action(obs)

    def get_attn(self, obs):
        return self.trainer.batch_attn(obs)

    def target_action(self, batch_obs):
        # print(batch_obs)
        return self.trainer.target_action(batch_obs)


    def _run(self):
        self.sum_batch_size = 0
        self.tot_batch = 0
        warned = False
        self.graph = graph = tf.Graph()
        # n_envs = len(self.env_conns)
        with graph.as_default():
            self.sess = sess = make_session(graph, self.num_cpu)
            with sess:
                self.build()
                sess.run(tf.variables_initializer(self.var_list))

                if self.load_dir is not None:
                    self.restore_weights(load_dir=self.load_dir)
                elif self.cached_weights is not None:
                    self.restore_trainable_weights(self.cached_weights)
                    del self.cached_weights

                if self.train:
                    self.save_weights([os.path.join(self.save_dir, "episode-0")])

                self.main_conn.send(None)
                self.main_conn.recv()
                attention_mode = self.attention_mode

                while True:
                    obs_batch = np.zeros(shape=(self.obs_batch_size, *self.obs_shape))
                    receiver = [None for _ in range(self.obs_batch_size)]

                    cnt = 0
                    while cnt < self.obs_batch_size:
                        try:
                            obs_batch[cnt], receiver[cnt] = self.obs_queue.get(block=True, timeout=self.timeout)
                            # print(receiver[cnt])
                            cnt += 1
                        except queue.Empty:
                            break

                    self.sum_batch_size += cnt
                    self.tot_batch += 1
                    if cnt > 0:
                        # print(cnt, receiver[:cnt])
                        action = self.action(obs_batch[:cnt])
                        if attention_mode:
                            good_attn, adv_attn = self.get_attn(obs_batch[:cnt])
                        for i in range(cnt):
                            if attention_mode:
                                self.env_conns[receiver[i]].send((action[i], good_attn[i], adv_attn[i]))
                            else:
                                self.env_conns[receiver[i]].send(action[i])

                    if self.save_event.is_set():
                        episode = self.main_conn.recv()
                        save_dirs = [self.save_dir]
                        if episode is not None:
                            save_dirs.append(os.path.join(self.save_dir, "episode-{}".format(episode)))
                        self.main_conn.send(self.save_weights(save_dirs))
                        self.main_conn.recv()

                    if self.update_event.is_set():
                        # if self.n == 40:
                        #     print("#{} start update".format(self.index))
                        if self.sum_batch_size / self.tot_batch / self.obs_batch_size < .5 and not warned:
                            warned = True
                            print("Batch load insufficient ({:.2%})! Consider higher timeout!".format(self.sum_batch_size / self.tot_batch / self.obs_batch_size))
                        sampled_index, data_length = self.main_conn.recv()

                        total_numbers = sum(self.obs_len) + sum(self.act_len) + self.n + sum(self.obs_len) + self.n
                        float_length = 4
                        assert total_numbers * float_length == data_length

                        obs_n = [np.zeros((self.batch_size, self.obs_len[i])) for i in range(self.n)]
                        action_n = [np.zeros((self.batch_size, self.act_len[i])) for i in range(self.n)]
                        reward = np.zeros(self.batch_size)
                        obs_next_n = [np.zeros((self.batch_size, self.obs_len[i])) for i in range(self.n)]
                        done = np.zeros(self.batch_size)

                        # print(sampled_index)
                        for i, index in enumerate(sampled_index):
                            self.tmp_file.seek(index * data_length)
                            flat_data = bytes_to_exp(self.tmp_file, total_numbers)

                            last = 0
                            for j in range(self.n):
                                l = self.obs_len[j]
                                obs_n[j][i], last = flat_data[last: last + l], last + l

                            for j in range(self.n):
                                l = self.act_len[j]
                                action_n[j][i], last = flat_data[last: last + l], last + l

                            reward[i] = flat_data[last + self.index]
                            last += self.n

                            for j in range(self.n):
                                l = self.obs_len[j]
                                obs_next_n[j][i], last = flat_data[last: last + l], last + l

                            done[i] = flat_data[last + self.index]
                            assert last + self.n == total_numbers

                        target_obs = []
                        for i in range(self.n):
                            if i < self.index:
                                target_obs.append(self.agent_recvs[i].recv())
                                self.agent_sends[i].send(obs_next_n[i])
                            elif i == self.index:
                                target_obs.append(obs_next_n[i])
                            else:
                                self.agent_sends[i].send(obs_next_n[i])
                                target_obs.append(self.agent_recvs[i].recv())

                        target_actions = self.target_action(np.concatenate(target_obs, axis=0))

                        # print(target_actions.shape)

                        target_action_n = [None for _ in range(self.n)]
                        for i in range(self.n):
                            ta = target_actions[i * self.batch_size: (i + 1) * self.batch_size]
                            if i < self.index:
                                target_action_n[i] = self.agent_recvs[i].recv()
                                self.agent_sends[i].send(ta)
                            elif i == self.index:
                                target_action_n[i] = ta
                            else:
                                self.agent_sends[i].send(ta)
                                target_action_n[i] = self.agent_recvs[i].recv()

                        if self.n == 40:
                            # print("#{} target action done".format(self.index))
                            self.main_conn.recv()

                        self.update(((obs_n, action_n, reward, obs_next_n, done), target_action_n))

                        if self.n == 40:
                            gc.collect()


                        self.main_conn.send(None)
                        self.main_conn.recv()

    def run(self):
        # self.server = tf.train.Server(CLUSTER, job_name="local%d" % self.index, task_index=0)

        if self.use_gpu:
            # print("GPU")
            with tf.device("/device:GPU:%d" % self.index):
                self._run()
        else:
            with tf.device("/cpu:0"):
                self._run()

    def save_weights(self, save_dirs):
        import joblib

        all_vars = self.sess.run(self.var_list)
        all_save_dict = {v.name: value for v, value in zip(self.var_list, all_vars)}
        trainable_save_dict = {name_encode(v.name, convert=False)[0]: all_save_dict[v.name]
                               for v in self.trainable_var_list}
        for save_dir in save_dirs:
            all_save_path = os.path.join(save_dir, "agent{}.weights".format(self.index))
            touch_path(all_save_path)
            # print(len(save_dict))
            joblib.dump(all_save_dict, all_save_path)

            trainable_save_path = os.path.join(save_dir, "agent{}.trainable-weights".format(self.index))
            touch_path(trainable_save_path)
            joblib.dump(trainable_save_dict, trainable_save_path)

        return trainable_save_dict

    def preupdate(self):
        self.trainer.preupdate()
        return None

    def update(self, args):
        data, target_act_next_n = args
        # _obs_n, _action_n, _rewards, _obs_next_n, _dones = data
        self.trainer.update(data, target_act_next_n)
        # return [0.] * 5


class Environment(multiprocessing.Process):
    def __init__(self, env, index, max_len, actor_queues, actor_conns, main_conn, experience_queue, attention_mode):
        multiprocessing.Process.__init__(self, daemon=True)

        self.env = env
        self.index = index
        self.n = env.n
        self.max_len = max_len
        self.actor_queues = actor_queues
        self.actor_conns = actor_conns
        self.main_conn = main_conn
        self.experience_queue = experience_queue
        self.attention_mode = attention_mode

    def run(self):
        env, n, actor_queues, actor_conns, experience_queue = \
            self.env, self.n, self.actor_queues, self.actor_conns, self.experience_queue

        attention_mode = self.attention_mode
        while True:
            obs_n = env.reset()
            steps = 0
            sum_reward_n = [0.] * n
            while True:
                for i in range(n):
                    actor_queues[i].put((obs_n[i], self.index))

                action_n = []
                good_attn_n = []
                adv_attn_n = []
                for i in range(n):
                    recv = actor_conns[i].recv()
                    if attention_mode:
                        action, good_attn, adv_attn = recv
                        good_attn_n.append(good_attn)
                        adv_attn_n.append(adv_attn)
                    else:
                        action = recv
                    action_n.append(action)

                new_obs_n, reward_n, done_n, info_n = env.step(action_n)
                for i in range(n):
                    sum_reward_n[i] += reward_n[i]
                    reward_n[i] = np.array(reward_n[i])
                    done_n[i] = np.array(done_n[i])

                steps += 1
                end_of_episode = steps > self.max_len or all(done_n)
                experience_queue.put([self.index, obs_n, action_n, reward_n, new_obs_n, done_n, end_of_episode, sum_reward_n, info_n, good_attn_n, adv_attn_n])

                if end_of_episode:
                    num_episodes = self.main_conn.recv()
                    # print("num_episodes:", num_episodes)
                    if num_episodes is not None:
                        # print("Saving gif!")
                        memory = env.export_memory()
                        # print(self.index, len(memory), num_episodes)
                        joblib.dump(memory, os.path.join(FLAGS.save_dir, "episode-{}.gif-data".format(num_episodes)))
                        # print("gif saved.", self.index)
                        self.main_conn.send(None)

                if True:
                    pause = self.main_conn.recv()
                    if pause:
                        self.main_conn.recv()
                    if end_of_episode:
                        break
                    else:
                        obs_n = new_obs_n


def train(arglist, init_weight_config=None, cached_weights=None):
    global FLAGS, CACHED_WEIGHTS
    if cached_weights is not None:
        CACHED_WEIGHTS = cached_weights
    FLAGS = arglist
    # print(FLAGS.save_summary)

    os.makedirs(FLAGS.save_dir, exist_ok=True)

    assert FLAGS.train_rate % FLAGS.n_envs == 0 and FLAGS.save_rate % FLAGS.n_envs == 0

    frames = []

    n = FLAGS.num_adversaries + FLAGS.num_good

    old_n = None
    id_mapping = None
    old_load_dir = None
    curriculum = init_weight_config is not None
    if curriculum:  # curriculum
        global NEW_IDS, PERTURBATION
        old_n_good = init_weight_config["old_n_good"]
        old_n_adv = init_weight_config["old_n_adv"]
        old_n = old_n_good + old_n_adv
        id_mapping = init_weight_config["id_mapping"]
        old_load_dir = init_weight_config["old_load_dir"]

        if old_load_dir is not None:
            if type(old_load_dir) != list:
                old_load_dir = [old_load_dir] * old_n
            elif len(old_load_dir) == 2:
                # print("!!!")
                old_load_dir = [old_load_dir[0]] * old_n_adv + [old_load_dir[1]] * old_n_good
                # print(load_dir)
            elif len(old_load_dir) != old_n:
                raise Exception("Initial load dir number mismatch!")

        try:
            NEW_IDS = init_weight_config["new_ids"]
        except KeyError:
            NEW_IDS = []

        try:
            PERTURBATION = init_weight_config["perturbation"]
        except KeyError:
            PERTURBATION = None

        # print(len(CACHED_WEIGHTS))
        if old_load_dir is not None:
            load_all_weights(old_load_dir, old_n)

    register_environment(n_good=FLAGS.num_good, n_adv=FLAGS.num_adversaries, n_landmarks=0, n_food=FLAGS.num_food,
                         n_forests=0, init_weights=curriculum, id_mapping=id_mapping)
    # register_fc(init_fully_connected)

    n_envs = min(FLAGS.n_envs, FLAGS.num_episodes)
    env = make_env(FLAGS.scenario, FLAGS, FLAGS.benchmark)
    assert n == N_ADV + N_GOOD

    if FLAGS.render_gif:
        import joblib
        for episode in range(FLAGS.num_episodes):
            gif_data = joblib.load(os.path.join(FLAGS.load_dir, "episode-{}.gif-data".format(episode)))
            frames = env.render_from_memory(gif_data, mode='rgb_array')
            imageio.mimsave(os.path.join(FLAGS.save_dir, "episode-{}.gif".format(episode)),
                            frames, duration=1 / 5)
        return

    obs_shape_n = [env.observation_space[i].shape for i in range(n)]
    action_shape_n = [env.action_space[i].n for i in range(n)]
    num_adversaries = min(n, FLAGS.num_adversaries)

    agents = []
    envs = []

    agent_env_conns = [[multiprocessing.Pipe() for _ in range(n_envs)] for _ in range(n)]
    main_agent_conns = [multiprocessing.Pipe() for _ in range(n)]
    main_env_conns = [multiprocessing.Pipe() for _ in range(n_envs)]
    agent_agent_conns = [[multiprocessing.Pipe() for _ in range(n)] for _ in range(n)]
    experience_queue = multiprocessing.Queue()
    obs_queues = [multiprocessing.Queue() for _ in range(n)]
    update_event = multiprocessing.Event()
    save_event = multiprocessing.Event()

    update_cnt = 0

    assert GOOD_SHARE_WEIGHTS == False
    assert ADV_SHARE_WEIGHTS == False

    def pick(weights, i):
        ret = {}
        for v, w in weights.items():
            if int(v.split('/')[0]) == i:
                ret[v] = w
        return ret

    obs_len = [obs_shape_n[i][0] for i in range(n)]
    act_len = [action_shape_n[i] for i in range(n)]

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_file_name = os.path.join(tmp_dir, "buffer")
        tmp_file = open(tmp_file_name, "w+b")

        for i in range(num_adversaries):
            agents.append(Agent(index=i,
                                n=n,
                                obs_batch_size=FLAGS.n_envs,
                                obs_shape=obs_shape_n[i],
                                update_event=update_event,
                                save_event=save_event,
                                save_dir=FLAGS.save_dir,
                                load_dir=FLAGS.load_dir if FLAGS.restore else None,
                                cached_weights=CACHED_WEIGHTS if len(CACHED_WEIGHTS) > 0 else None,
                                main_conn=main_agent_conns[i][1],
                                obs_queue=obs_queues[i],
                                env_conns=[agent_env_conns[i][j][0] for j in range(n_envs)],
                                num_cpu=FLAGS.n_cpu_per_agent,
                                get_trainer=partial(get_adv_trainer, i=i, scope="adv{}".format(i),
                                                    env=env, obs_shape_n=obs_shape_n),
                                use_gpu=FLAGS.use_gpu,
                                timeout=FLAGS.timeout,
                                attention_mode=FLAGS.show_attention,
                                agent_sends=[agent_agent_conns[i][j][0] for j in range(n)],
                                agent_recvs=[agent_agent_conns[j][i][1] for j in range(n)],
                                tmp_file=tmp_file_name,
                                obs_len=obs_len,
                                act_len=act_len,
                                batch_size=FLAGS.batch_size,
                                train=FLAGS.train_rate > 0
                                ))
        # print(num_adversaries, n)
        for i in range(num_adversaries, n):
            agents.append(Agent(index=i,
                                n=n,
                                obs_batch_size=FLAGS.n_envs,
                                obs_shape=obs_shape_n[i],
                                update_event=update_event,
                                save_event=save_event,
                                save_dir=FLAGS.save_dir,
                                load_dir=FLAGS.load_dir if FLAGS.restore else None,
                                cached_weights=CACHED_WEIGHTS if len(CACHED_WEIGHTS) > 0 else None,
                                main_conn=main_agent_conns[i][1],
                                obs_queue=obs_queues[i],
                                env_conns=[agent_env_conns[i][j][0] for j in range(n_envs)],
                                num_cpu=FLAGS.n_cpu_per_agent,
                                get_trainer=partial(get_good_trainer, i=i, scope="good{}".format(i - num_adversaries),
                                                    env=env, obs_shape_n=obs_shape_n),
                                use_gpu=FLAGS.use_gpu,
                                timeout=FLAGS.timeout,
                                attention_mode=FLAGS.show_attention,
                                agent_sends=[agent_agent_conns[i][j][0] for j in range(n)],
                                agent_recvs=[agent_agent_conns[j][i][1] for j in range(n)],
                                tmp_file=tmp_file_name,
                                obs_len=obs_len,
                                act_len=act_len,
                                batch_size=FLAGS.batch_size,
                                train=FLAGS.train_rate > 0
                                ))

        for i in range(n_envs):
            envs.append(Environment(env=make_env(FLAGS.scenario, FLAGS, FLAGS.benchmark),
                                    index=i,
                                    max_len=FLAGS.max_episode_len,
                                    actor_queues=obs_queues,
                                    actor_conns=[agent_env_conns[j][i][1] for j in range(n)],
                                    main_conn=main_env_conns[i][1],
                                    experience_queue=experience_queue,
                                    attention_mode=FLAGS.show_attention))

        tmp0 = time.time()
        print("Starting building graph & initialization & restoring...")

        for i in range(n):
            agents[i].start()
        for i in range(n):
            main_agent_conns[i][0].recv()

        print("Building graph & initialization & restoring done in {0:.2f} seconds".format(time.time() - tmp0))

        for i in range(n_envs):
            envs[i].start()

        for i in range(n):
            main_agent_conns[i][0].send(None)

        action_time = 0.0
        target_action_time = 0.0
        env_time = 0.0
        experience_time = 0.0
        sample_time = 0.0
        preupdate_time = 0.0
        update_time = 0.0
        pure_update_time = 0.0
        real_train_time = [0.0] * 5

        # obs_n = [env[i].reset() for i in range(n_envs)]
        episode_step = 0
        num_episodes = 0
        train_step = 0

        t_start = time.time()

        agent_rewards = [[] for _ in range(n)]

        train_start_time = time.time()
        print("Start training!")

        sess = tf.Session()

        last_update_time = time.time()

        with sess.as_default():
            with tf.name_scope('summaries'):
                reward_phs = []
                for i in range(FLAGS.num_adversaries):
                    with tf.name_scope('adv%d' % i):
                        reward_ph = tf.placeholder(dtype=tf.float32, shape=[], name='reward')
                        tf.summary.scalar('reward', reward_ph)
                        reward_phs.append(reward_ph)
                for i in range(FLAGS.num_adversaries, n):
                    with tf.name_scope('good%d' % (i - FLAGS.num_adversaries)):
                        reward_ph = tf.placeholder(dtype=tf.float32, shape=[], name='reward')
                        tf.summary.scalar('reward', reward_ph)
                        reward_phs.append(reward_ph)
                # episode_reward = tf.placeholder(dtype=tf.float32, shape=[], name='episode_reward')
                # tf.summary.scalar('episode_reward', episode_reward)
            summaries = tf.summary.merge_all()
            writer = tf.summary.FileWriter(FLAGS.save_dir + '/summaries', sess.graph)
            tf.variables_initializer(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='summaries')).run()

        num_steps_ahead = n_envs
        num_steps = 0

        time_grass = [np.zeros(n) for _ in range(n_envs)]
        time_live = [np.zeros(n) for _ in range(n_envs)]

        good_attn = [[] for _ in range(n_envs)]
        adv_attn = [[] for _ in range(n_envs)]

        time_grass_all = []
        time_live_all = []

        exp_len = None
        BUFFER_SIZE = int(1e6)
        buffer_len = 0

        test_tmp_file = open(tmp_file_name, "rb")

        record = n < 40 or FLAGS.train_rate == 0

        t_ep = time.time()
        while True:
            cnt = 0
            stop = False
            while cnt < n_envs:
                index, obs_n, action_n, reward_n, new_obs_n, done_n, end_of_episode, sum_reward_n, info_n, good_attn_n, adv_attn_n = experience_queue.get()
                if record:
                    good_attn[index].append(list(map(lambda x: x.tolist(), good_attn_n)))
                    adv_attn[index].append(list(map(lambda x: x.tolist(), adv_attn_n)))
                # print(good_attn_n, adv_attn_n)
                # print(info_n)
                if record:
                    for i in range(n):
                        time_grass[index][i] += info_n[i][0]
                        time_live[index][i] = info_n[i][1]
                if FLAGS.train_rate > 0:
                    # replay_buffer.add([obs_n, action_n, reward_n, new_obs_n, done_n])
                    exp_len, flat_data = exp_to_bytes(tmp_file, [obs_n, action_n, reward_n, new_obs_n, done_n])

                    tmp_file.flush()

                    buffer_len += 1
                    if buffer_len % BUFFER_SIZE == 0:
                        tmp_file.seek(0)

                if True:
                    num_steps += 1
                    if end_of_episode:
                        num_episodes += 1

                        if record:
                            time_grass_all.append(time_grass[index])
                            time_live_all.append(time_live[index])
                            time_grass[index] = np.zeros(n)
                            time_live[index] = np.zeros(n)

                            for i in range(n):
                                agent_rewards[i].append(sum_reward_n[i])

                        if FLAGS.show_attention:
                            import joblib
                            joblib.dump((good_attn[index], adv_attn[index]), os.path.join(FLAGS.save_dir, "episode-{}.attn".format(num_episodes - 1)))

                        if record:
                            good_attn[index] = []
                            adv_attn[index] = []

                        # episode_rewards.append(sum(sum_reward_n[i]))
                        if FLAGS.save_gif_data:
                            main_env_conns[index][0].send(num_episodes - 1)
                            main_env_conns[index][0].recv()
                        else:
                            main_env_conns[index][0].send(None)

                        # print(FLAGS.save_summary)
                        if FLAGS.save_summary:
                            summary = sess.run([summaries],
                                               feed_dict=dict([(reward_phs[i], agent_rewards[i][-1]) for i in range(n)])
                                               )[0]
                            writer.add_summary(summary, num_episodes)

                        if FLAGS.save_rate > 0 and num_episodes % FLAGS.save_rate == 0:
                            tmp0 = time.time()
                            save_event.set()
                            for i in range(n):
                                if FLAGS.checkpoint_rate > 0 and num_episodes % FLAGS.checkpoint_rate == 0:
                                    main_agent_conns[i][0].send(num_episodes)
                                else:
                                    main_agent_conns[i][0].send(None)
                            for i in range(n):
                                CACHED_WEIGHTS.update(main_agent_conns[i][0].recv())

                            save_event.clear()
                            for i in range(n):
                                main_agent_conns[i][0].send(None)

                            # replay_buffer.save(FLAGS.save_dir)
                            with open(os.path.join(FLAGS.save_dir, "progress"), "w") as f:
                                f.write(str(num_episodes))

                            save_time = time.time() - tmp0

                            this_time = time.time() - t_ep
                            print(
                                "{} episodes, total time: {:.2f}, update time: {:.2f} (sample time: {:.2f}, target action time: {:.2f}, pure update time: {:.2f}), save time: {:.2f}.".format(
                                    num_episodes, this_time, update_time, sample_time, target_action_time,
                                    pure_update_time, save_time))
                            print("Estimated finish time: {}".format(format_time((FLAGS.num_episodes - num_episodes) / FLAGS.save_rate * this_time)))
                            if record:
                                print("Time grass: {}, time live: {}".format(np.mean(np.array(time_grass_all[-20:]), axis=0), np.mean(np.array(time_live_all[-20:]), axis=0)))
                                print("reward:", [np.mean(np.array(agent_rewards[i][-20:])) for i in range(n)])
                            t_ep = time.time()
                            update_time = 0.
                            sample_time = 0.
                            target_action_time = 0.
                            pure_update_time = 0.

                        if num_episodes >= FLAGS.num_episodes:
                            stop = True
                            break
                    # print(num_steps_ahead, len(replay_buffer))
                    if FLAGS.train_rate > 0 and num_steps_ahead % FLAGS.train_rate == 0 and buffer_len >= FLAGS.batch_size * FLAGS.max_episode_len:
                        # print("Ready to train for env {}".format(i))
                        main_env_conns[index][0].send(True)
                        cnt += 1
                    else:
                        main_env_conns[index][0].send(False)
                        num_steps_ahead += 1

            if stop:
                break

            assert num_steps == num_steps_ahead

            # print(num_steps)

            # print(num_episodes)
            t_update = time.time()
            tmp_file.flush()
            update_event.set()

            tmp0 = time.time()
            sample_time += time.time() - tmp0
            tmp0 = time.time()
            for i in range(n):
                indices = random.sample(range(min(buffer_len, BUFFER_SIZE)), FLAGS.batch_size)
                main_agent_conns[i][0].send((indices, exp_len))

            if n == 40:
                indices = list(range(n))
                K = 10
                while len(indices) > 0:
                    cur_indices, indices = indices[:K], indices[K:]
                    for i in cur_indices:
                        main_agent_conns[i][0].send(None)
                    for i in cur_indices:
                        main_agent_conns[i][0].recv()
            else:
                for i in range(n):
                    main_agent_conns[i][0].recv()

            pure_update_time += time.time() - tmp0

            if n == 40:
                update_cnt += 1
                print("update #{} done. ({:.2f})s".format(update_cnt, time.time() - last_update_time))
                last_update_time = time.time()
                gc.collect()
            update_event.clear()

            for i in range(n):
                main_agent_conns[i][0].send(None)

            for i in range(n_envs):
                main_env_conns[i][0].send(None)

            num_steps_ahead += n_envs

            update_time += time.time() - t_update

    for i in range(n):
        agents[i].terminate()
        agents[i].join()

    for i in range(n_envs):
        envs[i].terminate()
        envs[i].join()

    # print(len(time_grass_all), time_grass_all[0])
    print("Time grass: {}, time live: {}".format(np.mean(np.array(time_grass_all), axis=0),
                                                 np.mean(np.array(time_live_all), axis=0)))

    # for ar in agent_rewards:
    #     print(np.mean(np.array(ar)))
    print("Total training time: {}.".format(format_time(time.time() - t_start)))

    return {
            "cached_weights": CACHED_WEIGHTS,
            "agent_rewards": agent_rewards,
            "time_grass": np.array(time_grass_all).T,
            "time_live":  np.array(time_live_all).T
        }
