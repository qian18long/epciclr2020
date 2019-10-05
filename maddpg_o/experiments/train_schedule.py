from .train_helper.train_helpers import parse_args
from .train_helper.proxy_train import proxy_train
import os
import random


class Schedule(object):
    def __init__(self, method=None, num_episodes=None, perturbation=None):
        self.method = method
        self.num_episodes = num_episodes
        self.perturbation = perturbation


DUMMY_SCHEDULE = [Schedule()]

TEST_SCHEDULE = [Schedule(),
                 Schedule("add_sheep")]

CRAZY_SCHEDULE = [Schedule(),
                  Schedule("add_sheep"),
                  Schedule("add_wolf"),
                  Schedule("x2")]

STANDARD_SCHEDULE = [Schedule(),
                     Schedule("x2"),
                     Schedule("x2", num_episodes=50000)]

SWAP_SCHEDULE = [Schedule("swap")]


X2_SCHEDULE = [Schedule("x2")]

X2_PERT_SCHEDULE = [Schedule("x2", perturbation=.1)]

SCHEDULES = {
    "dummy": DUMMY_SCHEDULE,
    "test": TEST_SCHEDULE,
    "crazy": CRAZY_SCHEDULE,
    "x2": X2_SCHEDULE,
    "x2_pert": X2_PERT_SCHEDULE,
    "x1.5": [Schedule("x1.5")],
    "standard": STANDARD_SCHEDULE,
    "swap": SWAP_SCHEDULE,
}


def gen_id_mapping(old_n, new_n):
    ol = list(range(old_n))
    return sum([ol] * (new_n // old_n), []) + sorted(random.sample(ol, new_n % old_n))


def curriculum_train(arglist, train_schedule, initial_load_dir=None):
    import copy
    original_arglist = copy.deepcopy(arglist)
    n_phases = len(train_schedule)
    n_sheep = arglist.num_good
    n_wolves = arglist.num_adversaries
    root_save_dir = arglist.save_dir
    cached_weights = None

    for phase in range(n_phases):
        schedule = train_schedule[phase]
        print("Training phase %d starts" % phase)

        arglist.num_episodes = num_episodes = schedule.num_episodes or original_arglist.num_episodes
        id_mapping = None

        if schedule.method is None:
            init_weight_config = None
        else:
            init_weight_config = {
                "old_n_good": n_sheep,
                "old_n_adv": n_wolves,
                "new_ids": []
            }
            if schedule.method == "x2":
                n_sheep *= 2
                n_wolves *= 2
                id_mapping = list(range(n_sheep + n_wolves))
                for i in range(n_wolves):
                    id_mapping[i] = i % (n_wolves // 2)
                    if i >= n_wolves // 2:
                        init_weight_config["new_ids"].append(i)
                for i in range(n_wolves, n_sheep + n_wolves):
                    id_mapping[i] = (i - n_wolves) % (
                                n_sheep // 2) + n_wolves // 2
                    if i - n_wolves >= n_sheep // 2:
                        init_weight_config["new_ids"].append(i)
            elif schedule.method == "x1.5":
                new_sheep = n_sheep // 2
                new_wolves = n_wolves // 2
                id_mapping = list(range(n_wolves + new_wolves + n_sheep + new_sheep))
                for i in range(n_wolves):
                    id_mapping[i] = i
                for i in range(n_wolves, n_wolves + new_wolves):
                    id_mapping[i] = i - n_wolves
                    init_weight_config["new_ids"].append(i)
                for i in range(n_wolves + new_wolves, n_wolves + new_wolves + n_sheep):
                    id_mapping[i] = i - new_wolves
                for i in range(n_wolves + new_wolves + n_sheep, n_wolves + new_wolves + n_sheep + new_sheep):
                    id_mapping[i] = i - new_wolves - n_sheep
                    init_weight_config["new_ids"].append(i)
                n_sheep += new_sheep
                n_wolves += new_wolves
            elif schedule.method == "add_sheep":
                old_id = random.randrange(n_sheep) + n_wolves
                print("Copied sheep %d" % (old_id - n_wolves))
                n_sheep += 1
                id_mapping = list(range(n_sheep + n_wolves))
                id_mapping[n_wolves + n_sheep - 1] = old_id
                init_weight_config["new_ids"].append(n_wolves + n_sheep - 1)
            elif schedule.method == "add_wolf":
                old_id = random.randrange(n_wolves)
                print("Copied wolf %d" % old_id)
                n_wolves += 1
                id_mapping = list(range(n_sheep + n_wolves))
                id_mapping[n_wolves - 1] = old_id
                for i in range(n_wolves, n_sheep + n_wolves):
                    id_mapping[i] = i - 1
                init_weight_config["new_ids"].append(n_wolves - 1)
            elif schedule.method == "swap":
                id_mapping = list(range(n_wolves + n_sheep))
            elif type(schedule.method) == tuple:
                new_n_sheep, new_n_wolves = schedule.method
                sheep_id_mapping = gen_id_mapping(n_sheep, new_n_sheep)
                wolves_id_mapping = gen_id_mapping(n_wolves, new_n_wolves)
                id_mapping = []
                for i in range(new_n_wolves):
                    id_mapping[i] = wolves_id_mapping[i]

                for i in range(new_n_sheep):
                    id_mapping[i + new_n_wolves] = sheep_id_mapping[i] + n_wolves

                n_wolves, n_sheep = new_n_wolves, new_n_sheep
            else:
                raise NotImplementedError
            init_weight_config["old_load_dir"] = initial_load_dir if phase == 0 else None
            init_weight_config["id_mapping"] = id_mapping
            if schedule.perturbation is not None:
                init_weight_config["perturbation"] = schedule.perturbation

        print("Start training with %d sheep and %d wolves...\n" % (n_sheep, n_wolves))

        arglist.num_adversaries = n_wolves
        arglist.num_good = n_sheep
        arglist.save_dir = save_dir = os.path.join(root_save_dir, "phase-%d" % phase) if n_phases > 1 else root_save_dir
        arglist.save_rate = min(original_arglist.save_rate, num_episodes)

        cached_weights = proxy_train({"arglist": arglist,
                                      "init_weight_config": init_weight_config,
                                      "cached_weights":cached_weights})["cached_weights"]
        print("Cached weights: {}".format(len(cached_weights)))

        import json
        json.dump({
            "n_sheep": n_sheep,
            "n_wolves": n_wolves,
            "id_mapping": id_mapping,
            "num_episodes": num_episodes,
            "method": schedule.method
        }, open(os.path.join(save_dir, "train_schedule.json"), "w"))

        print("\nTraining phase %d ends.\n\n" % phase)


def make_gradual_x2(arglist):
    n_g = arglist.num_good
    n_a = arglist.num_adversaries
    n_ep = arglist.num_episodes // 2

    schedule = [Schedule(None, n_ep)]
    step = n_ep // (n_a + n_g)

    ratio = n_g / n_a
    for i in range(n_a + n_g):
        ratio_now = n_g / n_a
        if ratio_now < ratio - 1e-8:
            schedule.append(Schedule("add_sheep", step))
            n_g += 1
        elif ratio_now > ratio + 1e-8:
            schedule.append(Schedule("add_wolf", step))
            n_a += 1
        elif random.randrange(2) == 0:
            schedule.append(Schedule("add_sheep", step))
            n_g += 1
        else:
            schedule.append(Schedule("add_wolf", step))
            n_a += 1

    return schedule


def add_extra_flags(parser):
    parser.add_argument('--init-load-dir', type=str)
    parser.add_argument('--sheep-init-load-dir', type=str)
    parser.add_argument('--wolf-init-load-dir', type=str)
    parser.add_argument('--schedule', type=str)
    parser.add_argument('--new-n-wolves', type=int)
    parser.add_argument('--new-n-sheep', type=int)
    # parser.add_argument('--wolf-load-dirs', type=str, nargs='+')
    return parser


if __name__ == "__main__":
    arglist = parse_args(add_extra_flags)

    sheep_init_load_dir = arglist.sheep_init_load_dir or arglist.init_load_dir
    wolf_init_load_dir = arglist.wolf_init_load_dir or arglist.init_load_dir
    init_load_dir = None if sheep_init_load_dir is None or wolf_init_load_dir is None else [wolf_init_load_dir, sheep_init_load_dir]

    if arglist.schedule is not None:
        schedule = SCHEDULES[arglist.schedule]
    else:
        schedule = [Schedule((arglist.new_n_sheep, arglist.new_n_wolves))]

    curriculum_train(arglist, train_schedule=schedule, initial_load_dir=init_load_dir)