from maddpg_o.experiments.train_helper.train_helpers import parse_args, train

if __name__ == "__main__":
    arglist = parse_args()
    train(arglist)