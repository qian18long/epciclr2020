from .train_helper.train_helpers import parse_args
from .train_helper.proxy_train import proxy_train
import os
import joblib
import random


def add_extra_flags(parser):
    parser.add_argument('--init-load-dir', type=str)
    parser.add_argument('--sheep-init-load-dir', type=str)
    parser.add_argument('--wolf-init-load-dir', type=str)
    parser.add_argument('--new-n-wolves', type=int)
    parser.add_argument('--new-n-sheep', type=int)
    # parser.add_argument('--wolf-load-dirs', type=str, nargs='+')
    return parser


FLAGS = None


def touch_dir(dirname):
    os.makedirs(dirname, exist_ok=True)


def read_and_renumber(path, i, j):
    weights = joblib.load(path)
    new_weights = dict()
    for name, v in weights.items():
        dname = name.split('/')
        assert dname[0] == str(i)
        dname[0] = str(j)
        new_weights['/'.join(dname)] = v
    return new_weights


def renumber(read_path, write_path, id_mapping, d0, d1):
    for i in range(len(id_mapping)):
        weights = read_and_renumber(os.path.join(read_path, "agent{}.trainable-weights".format(id_mapping[i] + d1)),
                                    id_mapping[i] + d1, i + d0)
        joblib.dump(weights, os.path.join(write_path, "agent{}.trainable-weights".format(i + d0)))


def gen_id_mapping(old_n, new_n):
    ol = list(range(old_n))
    return sum([ol] * (new_n // old_n), []) + sorted(random.sample(ol, new_n % old_n))


def mix_match():
    save_dir = FLAGS.save_dir
    touch_dir(save_dir)

    wolf_load_dir = FLAGS.wolf_init_load_dir or FLAGS.init_load_dir
    sheep_load_dir = FLAGS.sheep_init_load_dir or FLAGS.init_load_dir

    n_sheep = FLAGS.num_good
    n_wolves = FLAGS.num_adversaries

    new_n_sheep = FLAGS.new_n_sheep
    new_n_wolves = FLAGS.new_n_wolves

    sheep_id_mapping = gen_id_mapping(n_sheep, new_n_sheep)
    wolves_id_mapping = gen_id_mapping(n_wolves, new_n_wolves)

    renumber(wolf_load_dir, save_dir, wolves_id_mapping, 0, 0)
    renumber(sheep_load_dir, save_dir, sheep_id_mapping, new_n_wolves, n_wolves)


if __name__ == "__main__":
    FLAGS = parse_args(add_extra_flags)

    mix_match()
