from maddpg_o.experiments.train_helper.train_helpers import parse_args, train

def add_extra_flags(parser):
    parser.add_argument('--old-id', type=int)
    parser.add_argument('--new-id', type=int)
    parser.add_argument('--old-load-dir', type=str)
    return parser


if __name__ == "__main__":
    arglist = parse_args(add_extra_flags)
    id_mapping = list(range(arglist.num_good + arglist.num_adversaries))
    id_mapping[arglist.new_id] = arglist.old_id
    train(arglist, init_weight_config={
        "old_n": arglist.num_good + arglist.num_adversaries - 1,
        "id_mapping": id_mapping,
        "old_load_dir": arglist.old_load_dir
    })