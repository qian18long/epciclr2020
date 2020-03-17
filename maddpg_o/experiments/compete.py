from .train_helper.train_helpers import parse_args
from .train_helper.proxy_train import proxy_train
import numpy as np
import os


def add_extra_flags(parser):
    parser.add_argument('--competitor-load-dirs', type=str, nargs='+')
    parser.add_argument('--competitor-num-units', type=int, nargs='+')
    parser.add_argument('--competitor-checkpoint-rates', type=int, nargs='+')
    parser.add_argument('--competitor-num-episodes', type=int, nargs='+')
    parser.add_argument('--competitor-start-episodes', type=int, nargs='+')
    parser.add_argument('--competitor-checkpoint-suffices', type=str, nargs='+')
    parser.add_argument('--baseline-checkpoint-rates', type=int, nargs='+')
    parser.add_argument('--baseline-num-episodes', type=int, nargs='+')
    parser.add_argument('--baseline-start-episodes', type=int, nargs='+')
    parser.add_argument('--baseline-checkpoint-suffices', type=str, nargs='+')
    parser.add_argument('--competitor-models', type=str, nargs='+', default=["maddpg"])
    parser.add_argument('--competitor-share-weights', type=bool, nargs='+', default=[False])
    parser.add_argument('--baseline-load-dirs', type=str, nargs='+')
    parser.add_argument('--baseline-num-units', type=int, nargs='+')
    parser.add_argument('--baseline-models', type=str, nargs='+', default=["maddpg"])
    parser.add_argument('--baseline-share-weights', type=bool, nargs='+', default=[False])
    parser.add_argument('--parallel-limit', type=int, default=0)
    parser.add_argument('--symmetric', action="store_true", default=False)
    parser.add_argument('--antisymmetric', action="store_true", default=False)
    parser.add_argument('--dot-product', action="store_true", default=False)
    parser.add_argument('--double', action="store_true", default=False)
    # parser.add_argument('--wolf-load-dirs', type=str, nargs='+')
    return parser


def show_group_statistics(rewards, category):
    if len(rewards) == 0:
        return [], [], 0.0, 0.0
    print("-- {} --".format(category))
    rewards = np.array(rewards)
    print("Individuals:")
    print("mean:", np.mean(rewards, axis=1))
    print("var:", np.var(rewards, axis=1))

    sum_rewards = np.sum(rewards, axis=0)
    print("Sum:")
    print("mean:", np.mean(sum_rewards))
    print("var:", np.var(sum_rewards))

    return np.mean(rewards, axis=1), np.var(rewards, axis=1), np.mean(sum_rewards), np.var(sum_rewards)


def compete(arglist):
    import copy
    original_arglist = copy.deepcopy(arglist)

    dot_product = original_arglist.dot_product
    double = original_arglist.double
    # half = original_arglist.half

    competitor_load_dirs = arglist.competitor_load_dirs
    baseline_load_dirs = arglist.baseline_load_dirs
    competitor_models = arglist.competitor_models
    baseline_models = arglist.baseline_models
    competitor_share_weights = arglist.competitor_share_weights
    baseline_share_weights = arglist.baseline_share_weights
    competitor_num_units = arglist.competitor_num_units or [arglist.num_units]
    baseline_num_units = arglist.baseline_num_units or [arglist.num_units]

    n_competitors = len(competitor_load_dirs)
    n_baselines = len(baseline_load_dirs)

    def expand(arr, l):
        if len(arr) < l:
            return [arr[0]] * l
        else:
            return arr

    competitor_models = expand(competitor_models, n_competitors)
    baseline_models = expand(baseline_models, n_baselines)
    competitor_num_units = expand(competitor_num_units, n_competitors)
    baseline_num_units = expand(baseline_num_units, n_baselines)
    competitor_share_weights = expand(competitor_share_weights, n_competitors)
    baseline_share_weights = expand(baseline_share_weights, n_baselines)

    symmetric = arglist.symmetric
    antisymmetric = arglist.antisymmetric

    if arglist.competitor_checkpoint_rates is not None:
        competitor_checkpoint_rates = expand(arglist.competitor_checkpoint_rates, n_competitors)
        competitor_start_episodes = expand(arglist.competitor_start_episodes or competitor_checkpoint_rates, n_competitors)
        competitor_checkpoint_suffices = expand(arglist.competitor_checkpoint_suffices or ["episode-"], n_competitors)
        competitor_num_episodes = expand(arglist.competitor_num_episodes, n_competitors)
        new_competitor_load_dirs = []
        new_competitor_share_weights = []
        new_competitor_models = []
        new_competitor_num_units = []
        for i in range(n_competitors):
            rate = competitor_checkpoint_rates[i]
            n_ep = competitor_num_episodes[i]
            start = competitor_start_episodes[i]
            suffix = competitor_checkpoint_suffices[i]
            for j in range(start, n_ep + 1, rate):
                new_competitor_load_dirs.append(os.path.join(competitor_load_dirs[i], "{}{}".format(suffix, j)))
                new_competitor_models.append(competitor_models[i])
                new_competitor_share_weights.append(competitor_share_weights[i])
                new_competitor_num_units.append(competitor_num_units[i])
        competitor_load_dirs = new_competitor_load_dirs
        competitor_models = new_competitor_models
        competitor_share_weights = new_competitor_share_weights
        competitor_num_units = new_competitor_num_units
        n_competitors = len(competitor_load_dirs)

    if arglist.baseline_checkpoint_rates is not None:
        baseline_checkpoint_rates = expand(arglist.baseline_checkpoint_rates, n_baselines)
        baseline_start_episodes = expand(arglist.baseline_start_episodes or baseline_checkpoint_rates, n_baselines)
        baseline_checkpoint_suffices = expand(arglist.baseline_checkpoint_suffices or ["episode-"], n_baselines)
        baseline_num_episodes = expand(arglist.baseline_num_episodes, n_baselines)
        new_baseline_load_dirs = []
        new_baseline_share_weights = []
        new_baseline_models = []
        new_baseline_num_units = []
        for i in range(n_baselines):
            rate = baseline_checkpoint_rates[i]
            n_ep = baseline_num_episodes[i]
            start = baseline_start_episodes[i]
            suffix = baseline_checkpoint_suffices[i]
            for j in range(start, n_ep + 1, rate):
                new_baseline_load_dirs.append(os.path.join(baseline_load_dirs[i], "{}{}".format(suffix, j)))
                new_baseline_models.append(baseline_models[i])
                new_baseline_share_weights.append(baseline_share_weights[i])
                new_baseline_num_units.append(baseline_num_units[i])
        baseline_load_dirs = new_baseline_load_dirs
        baseline_models = new_baseline_models
        baseline_share_weights = new_baseline_share_weights
        baseline_num_units = new_baseline_num_units
        n_baselines = len(baseline_load_dirs)

    n_sheep = original_arglist.num_good
    n_wolves = original_arglist.num_adversaries
    n = n_sheep + n_wolves

    arglist.train_rate = 0
    arglist.save_rate = 0
    arglist.save_summary = False

    init_weight_config = {
        "old_n_good": n_sheep,
        "old_n_adv": n_wolves,
        "new_ids": []
    }

    if double:
        n_sheep *= 2
        n_wolves *= 2
        arglist.num_adversaries *= 2
        arglist.num_good *= 2
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
        init_weight_config["id_mapping"] = id_mapping
    else:
        init_weight_config["id_mapping"] = list(range(n))


    kwargs_list = []

    for i in range(n_competitors):
        for j in range(n_baselines):
            if dot_product and i != j:
                continue
            if not antisymmetric:
                new_arglist = copy.deepcopy(arglist)
                new_arglist.adv_policy = competitor_models[i]
                new_arglist.adv_share_weights = competitor_share_weights[i]
                new_arglist.adv_num_units = competitor_num_units[i]
                new_arglist.good_policy = baseline_models[j]
                new_arglist.good_share_weights = baseline_share_weights[j]
                new_arglist.good_num_units = baseline_num_units[j]
                new_arglist.save_dir = os.path.join(original_arglist.save_dir, "c{}xb{}".format(i, j)) if original_arglist.save_dir is not None else None
                new_init_weight_config = copy.deepcopy(init_weight_config)
                new_init_weight_config["old_load_dir"] = [competitor_load_dirs[i], baseline_load_dirs[j]]

                kwargs_list.append({"arglist": new_arglist, "init_weight_config": new_init_weight_config})

            if not symmetric:
                new_arglist = copy.deepcopy(arglist)
                new_arglist.good_policy = competitor_models[i]
                new_arglist.good_share_weights = competitor_share_weights[i]
                new_arglist.good_num_units = competitor_num_units[i]
                new_arglist.adv_policy = baseline_models[j]
                new_arglist.adv_share_weights = baseline_share_weights[j]
                new_arglist.adv_num_units = baseline_num_units[j]
                new_arglist.save_dir = os.path.join(original_arglist.save_dir, "b{}xc{}".format(j, i)) if original_arglist.save_dir is not None else None
                new_init_weight_config = copy.deepcopy(init_weight_config)
                new_init_weight_config["old_load_dir"] = [baseline_load_dirs[j], competitor_load_dirs[i]]

                kwargs_list.append({"arglist": new_arglist, "init_weight_config": new_init_weight_config})

    results = []
    limit = original_arglist.parallel_limit
    if limit == 0:
        limit = len(kwargs_list)

    while len(kwargs_list) > 0:
        results += proxy_train(kwargs_list[:limit])
        kwargs_list = kwargs_list[limit:]

    detailed_reports = []

    competitor_wolf_scores = np.zeros((n_competitors, 1 if dot_product else n_baselines))
    competitor_sheep_scores = np.zeros((n_competitors, 1 if dot_product else n_baselines))

    baseline_wolf_scores = np.zeros((n_baselines, 1 if dot_product else n_competitors))
    baseline_sheep_scores = np.zeros((n_baselines, 1 if dot_product else n_competitors))


    competitor_wolf_tg = np.zeros((n_competitors, 1 if dot_product else n_baselines))
    competitor_sheep_tg = np.zeros((n_competitors, 1 if dot_product else n_baselines))

    baseline_wolf_tg = np.zeros((n_baselines, 1 if dot_product else n_competitors))
    baseline_sheep_tg = np.zeros((n_baselines, 1 if dot_product else n_competitors))

    competitor_wolf_tl = np.zeros((n_competitors, 1 if dot_product else n_baselines))
    competitor_sheep_tl = np.zeros((n_competitors, 1 if dot_product else n_baselines))

    baseline_wolf_tl = np.zeros((n_baselines, 1 if dot_product else n_competitors))
    baseline_sheep_tl = np.zeros((n_baselines, 1 if dot_product else n_competitors))

    for i in range(n_competitors):
        for j in range(n_baselines):
            if dot_product and i != j:
                continue
            if not antisymmetric:
                agent_rewards = results[0]["agent_rewards"]
                time_grass = results[0]["time_grass"]
                time_live = results[0]["time_live"]
                results = results[1:]

                # print("ASDAS", agent_rewards, time_grass, time_live)

                print("\n\n-- Wolves from competitor {} V.S. Sheep from baseline {} ({} and {}) --".format(i, j, competitor_load_dirs[i], baseline_load_dirs[j]))
                print("\nWolves")
                ind_wolf_score, ind_wolf_var, wolf_score, wolf_var = show_group_statistics(agent_rewards[:n_wolves], "rewards")
                ind_wolf_score_tg, ind_wolf_var_tg, wolf_score_tg, wolf_var_tg = show_group_statistics(time_grass[:n_wolves], "time grass")
                ind_wolf_score_tl, ind_wolf_var_tl, wolf_score_tl, wolf_var_tl = show_group_statistics(time_live[:n_wolves], "time live")
                print("\nSheep")
                ind_sheep_score, ind_sheep_var, sheep_score, sheep_var = show_group_statistics(agent_rewards[n_wolves:], "rewards")
                ind_sheep_score_tg, ind_sheep_var_tg, sheep_score_tg, sheep_var_tg = show_group_statistics(time_grass[n_wolves:], "time grass")
                ind_sheep_score_tl, ind_sheep_var_tl, sheep_score_tl, sheep_var_tl = show_group_statistics(time_live[n_wolves:], "time live")

                print("wolf precision:", np.sqrt(wolf_var / arglist.num_episodes))
                print("sheep precision:", np.sqrt(sheep_var / arglist.num_episodes))

                competitor_wolf_scores[i][0 if dot_product else j] = wolf_score
                baseline_sheep_scores[j][0 if dot_product else i] = sheep_score

                competitor_wolf_tg[i][0 if dot_product else j] = wolf_score_tg / n_wolves if n_wolves > 0 else 0.
                baseline_sheep_tg[j][0 if dot_product else i] = sheep_score_tg / n_sheep

                competitor_wolf_tl[i][0 if dot_product else j] = wolf_score_tl / n_wolves if n_wolves > 0 else 0.
                baseline_sheep_tl[j][0 if dot_product else i] = sheep_score_tl / n_sheep

                report = {
                    "wolf": {
                        "id": i,
                        "side": "competitor",
                        "load_dir": competitor_load_dirs[i],
                        "model": competitor_models[i],
                        "sum_score": wolf_score,
                        "ind_score": ind_wolf_score.tolist(),
                        "avg_tg": wolf_score_tg / n_wolves,
                        "avg_tl": wolf_score_tl / n_wolves
                    } if n_wolves > 0 else {},
                    "sheep": {
                        "id": j,
                        "side": "baseline",
                        "load_dir": baseline_load_dirs[j],
                        "model": baseline_models[j],
                        "sum_score": sheep_score,
                        "ind_score": ind_sheep_score.tolist(),
                        "avg_tg": sheep_score_tg / n_sheep,
                        "avg_tl": sheep_score_tl / n_sheep
                    }
                }

                detailed_reports.append(report)

            if not symmetric:

                agent_rewards = results[0]["agent_rewards"]
                time_grass = results[0]["time_grass"]
                time_live = results[0]["time_live"]
                results = results[1:]

                print("\n\n-- Sheep from competitor {} V.S. Wolves from baseline {} ({} and {}) --".format(i, j,
                                                                                                               competitor_load_dirs[
                                                                                                               i],
                                                                                                           baseline_load_dirs[
                                                                                                               j]))
                print("\nWolves")
                ind_wolf_score, ind_wolf_var, wolf_score, wolf_var = show_group_statistics(agent_rewards[:n_wolves], "reward")
                ind_wolf_score_tg, ind_wolf_var_tg, wolf_score_tg, wolf_var_tg = show_group_statistics(time_grass[:n_wolves], "time grass")
                ind_wolf_score_tl, ind_wolf_var_tl, wolf_score_tl, wolf_var_tl = show_group_statistics(time_live[:n_wolves], "time live")
                print("\nSheep")
                ind_sheep_score, ind_sheep_var, sheep_score, sheep_var = show_group_statistics(agent_rewards[n_wolves:], "reward")
                ind_sheep_score_tg, ind_sheep_var_tg, sheep_score_tg, sheep_var_tg = show_group_statistics(time_grass[n_wolves:], "time grass")
                ind_sheep_score_tl, ind_sheep_var_tl, sheep_score_tl, sheep_var_tl = show_group_statistics(time_live[n_wolves:], "time live")

                print("wolf precision:", np.sqrt(wolf_var / arglist.num_episodes))
                print("sheep precision:", np.sqrt(sheep_var / arglist.num_episodes))

                competitor_sheep_scores[i][0 if dot_product else j] = sheep_score
                baseline_wolf_scores[j][0 if dot_product else i] = wolf_score

                competitor_sheep_tg[i][0 if dot_product else j] = sheep_score_tg / n_sheep
                baseline_wolf_tg[j][0 if dot_product else i] = wolf_score_tg / n_wolves if n_wolves > 0 else 0.

                competitor_sheep_tl[i][0 if dot_product else j] = sheep_score_tl / n_sheep
                baseline_wolf_tl[j][0 if dot_product else i] = wolf_score_tl / n_wolves if n_wolves > 0 else 0.
                baseline_wolf_tl[j][0 if dot_product else i] = wolf_score_tl / n_wolves if n_wolves > 0 else 0.

                report = {
                    "sheep": {
                        "id": i,
                        "side": "competitor",
                        "load_dir": competitor_load_dirs[i],
                        "model": competitor_models[i],
                        "sum_score": sheep_score,
                        "ind_score": ind_sheep_score.tolist(),
                        "avg_tg": sheep_score_tg / n_sheep,
                        "avg_tl": sheep_score_tl / n_sheep
                    },
                    "wolf": {
                        "id": j,
                        "side": "baseline",
                        "load_dir": baseline_load_dirs[j],
                        "model": baseline_models[j],
                        "sum_score": wolf_score,
                        "ind_score": ind_wolf_score.tolist(),
                        "avg_tg": wolf_score_tg / n_wolves,
                        "avg_tl": wolf_score_tl / n_wolves
                    } if n_wolves > 0 else {}
                }

                detailed_reports.append(report)

    print("\n\nCompetitor wolf scores:", competitor_wolf_scores)
    print("mean:", np.mean(competitor_wolf_scores, axis=-1))
    print("worst:", np.min(competitor_wolf_scores, axis=-1))
    print("\n\nCompetitor wolf time grass:", competitor_wolf_tg)
    print("mean:", np.mean(competitor_wolf_tg, axis=-1))
    print("worst:", np.min(competitor_wolf_tg, axis=-1))
    print("\n\nCompetitor wolf time live:", competitor_wolf_tl)
    print("mean:", np.mean(competitor_wolf_tl, axis=-1))
    print("worst:", np.min(competitor_wolf_tl, axis=-1))
    print("\nCompetitor sheep scores:", competitor_sheep_scores)
    print("mean:", np.mean(competitor_sheep_scores, axis=-1))
    print("worst:", np.min(competitor_sheep_scores, axis=-1))
    print("\n\nCompetitor sheep time grass:", competitor_sheep_tg)
    print("mean:", np.mean(competitor_sheep_tg, axis=-1))
    print("worst:", np.min(competitor_sheep_tg, axis=-1))
    print("\n\nCompetitor sheep time live:", competitor_sheep_tl)
    print("mean:", np.mean(competitor_sheep_tl, axis=-1))
    print("worst:", np.min(competitor_sheep_tl, axis=-1))

    print("\n\nBaseline wolf scores:", baseline_wolf_scores)
    print("mean:", np.mean(baseline_wolf_scores, axis=-1))
    print("worst:", np.min(baseline_wolf_scores, axis=-1))
    print("\n\nBaseline wolf time grass:", baseline_wolf_tg)
    print("mean:", np.mean(baseline_wolf_tg, axis=-1))
    print("worst:", np.min(baseline_wolf_tg, axis=-1))
    print("\n\nBaseline wolf time live:", baseline_wolf_tl)
    print("mean:", np.mean(baseline_wolf_tl, axis=-1))
    print("worst:", np.min(baseline_wolf_tl, axis=-1))
    print("\nBaseline sheep scores:", baseline_sheep_scores)
    print("mean:", np.mean(baseline_sheep_scores, axis=-1))
    print("worst:", np.min(baseline_sheep_scores, axis=-1))
    print("\n\nBaseline sheep time grass:", baseline_sheep_tg)
    print("mean:", np.mean(baseline_sheep_tg, axis=-1))
    print("worst:", np.min(baseline_sheep_tg, axis=-1))
    print("\n\nBaseline sheep time live:", baseline_sheep_tl)
    print("mean:", np.mean(baseline_sheep_tl, axis=-1))
    print("worst:", np.min(baseline_sheep_tl, axis=-1))

    report = {
        "competitor_groups": list(zip(competitor_load_dirs, competitor_models)),
        "baseline_groups": list(zip(baseline_load_dirs, baseline_models)),
        "competitor_wolf_scores": competitor_wolf_scores.tolist(),
        "competitor_wolf_tg": competitor_wolf_tg.tolist(),
        "competitor_wolf_tl": competitor_wolf_tl.tolist(),
        "competitor_sheep_scores": competitor_sheep_scores.tolist(),
        "competitor_sheep_tg": competitor_sheep_tg.tolist(),
        "competitor_sheep_tl": competitor_sheep_tl.tolist(),
        "baseline_wolf_scores": baseline_wolf_scores.tolist(),
        "baseline_wolf_tg": baseline_wolf_tg.tolist(),
        "baseline_wolf_tl": baseline_wolf_tl.tolist(),
        "baseline_sheep_scores": baseline_sheep_scores.tolist(),
        "baseline_sheep_tg": baseline_sheep_tg.tolist(),
        "baseline_sheep_tl": baseline_sheep_tl.tolist(),
        "detailed_reports": detailed_reports
    }

    if original_arglist.save_dir is not None:
        import json
        json.dump(report, open(os.path.join(original_arglist.save_dir, "report.json"), "w"))

    return report

if __name__ == "__main__":
    compete(parse_args(add_extra_flags))
