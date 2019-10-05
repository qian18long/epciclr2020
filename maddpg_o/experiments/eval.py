from maddpg_o.experiments.train_helper.train_helpers import parse_args, train

def add_extra_flags(parser):
    parser.add_argument('--sheep-load-dir', type=str)
    parser.add_argument('--wolf-load-dir', type=str)
    return parser

if __name__ == "__main__":
    arglist = parse_args(add_extra_flags)
    id_mapping = list(range(arglist.num_good + arglist.num_adversaries))
    train(arglist, init_weight_config={
        "old_n": arglist.num_good + arglist.num_adversaries,
        "id_mapping": id_mapping,
        "old_load_dir": [arglist.wolf_load_dir] * arglist.num_adversaries +
                        [arglist.sheep_load_dir] * arglist.num_good
    })