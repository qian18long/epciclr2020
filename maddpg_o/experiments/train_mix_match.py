from .train_helper.train_helpers import parse_args
from .train_helper.proxy_train import proxy_train
import os
import joblib


def add_extra_flags(parser):
    parser.add_argument('--sheep-init-load-dirs', nargs=2, type=str)
    parser.add_argument('--wolf-init-load-dirs', nargs=2, type=str)
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


def renumber(read_path, write_path, r, delta):
    for i in r:
        weights = read_and_renumber(os.path.join(read_path, "agent{}.trainable-weights".format(i)), i, i + delta)
        joblib.dump(weights, os.path.join(write_path, "agent{}.trainable-weights".format(i + delta)))


def mix_match(FLAGS):
    save_dir = FLAGS.save_dir
    initial_dir = os.path.join(save_dir, "initial")
    touch_dir(initial_dir)

    n_sheep = FLAGS.num_good
    n_wolves = FLAGS.num_adversaries

    renumber(FLAGS.wolf_init_load_dirs[0], initial_dir, range(n_wolves), 0)
    renumber(FLAGS.wolf_init_load_dirs[1], initial_dir, range(n_wolves), n_wolves)
    renumber(FLAGS.sheep_init_load_dirs[0], initial_dir, range(n_wolves, n_wolves + n_sheep), n_wolves)
    renumber(FLAGS.sheep_init_load_dirs[1], initial_dir, range(n_wolves, n_wolves + n_sheep), n_wolves + n_sheep)

    init_weight_config = {
        "old_n_good": n_sheep * 2,
        "old_n_adv": n_wolves * 2,
        "new_ids": [],
        "old_load_dir": initial_dir,
        "id_mapping": list(range(n_sheep * 2 + n_wolves * 2))
    }

    import copy
    flags = copy.deepcopy(FLAGS)

    flags.num_good = n_sheep * 2
    flags.num_adversaries = n_wolves * 2

    import json
    json.dump({
        "old_n_sheep": n_sheep,
        "old_n_wolves": n_wolves,
        "wolf_init_dirs": FLAGS.wolf_init_load_dirs,
        "sheep_init_dirs": FLAGS.sheep_init_load_dirs
    }, open(os.path.join(save_dir, "mix_match_config.json"), "w"))

    proxy_train({"arglist": flags,
                 "init_weight_config": init_weight_config})


if __name__ == "__main__":
    FLAGS = parse_args(add_extra_flags)

    mix_match(FLAGS)
