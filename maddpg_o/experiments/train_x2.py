from maddpg_o.experiments.train_helper.train_helpers import parse_args, train


def add_extra_flags(parser):
    parser.add_argument('--old-load-dir', type=str)
    return parser


if __name__ == "__main__":
    arglist = parse_args(add_extra_flags)
    id_mapping = list(range(arglist.num_good + arglist.num_adversaries))
    assert arglist.num_good % 2 == 0 and arglist.num_adversaries % 2 == 0
    for i in range(arglist.num_adversaries):
        id_mapping[i] = i % (arglist.num_adversaries // 2)
    for i in range(arglist.num_adversaries, arglist.num_good + arglist.num_adversaries):
        id_mapping[i] = (i - arglist.num_adversaries) % (arglist.num_good // 2) + arglist.num_adversaries // 2
    train(arglist, init_weight_config={
        "old_n": arglist.num_good // 2 + arglist.num_adversaries // 2,
        "id_mapping": id_mapping,
        "old_load_dir": arglist.old_load_dir
    })