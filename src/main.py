"""Main file"""


import argparse
import random
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

import runner
import wandb


@dataclass
class Config:
    """Config class for hyparms"""

    mutation_power: float
    generations: int
    num_agents: int
    top_limit: int
    n_runs: int
    n_survivors: int
    using_biases: bool
    seed: int


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation script")
    parser.add_argument("-s", "--seed", type=int, help="Seed", required=True)
    parser.add_argument("-l", "--layer", type=int, help="Layer", required=True)
    parser.add_argument("-e", "--experiment", type=str, help="Experiment", required=True)
    # parser.add_argument("-r", "--recomb", type=bool, help="Recombination", required=True)
    # parser.add_argument("-m", "--method", type=str, help="Recombination Method", required=True)

    args = parser.parse_args()

    recomb = False
    method = "random"

    if args.experiment == "muta":
        recomb = False
    if args.experiment == "recomb_random":
        recomb = True
        method = "random"
    if args.experiment == "recomb_layerwise":
        recomb = True
        method = "layerwise"
    return args.seed, args.layer, recomb, method


def main(seed: int, layer: int, recomb: bool, recomb_meth: str):
    """Main entrypoint"""

    evo_meth = ""

    if recomb:
        evo_meth = f"muta-recomb-{recomb_meth}"
    else:
        evo_meth = "muta"

    print(f"Data-{layer}-{evo_meth}-{seed}.csv")

    wandb.init(
        project="maqrl-eo",
        entity="mdsg",
        config={
            "mutation_power": 0.01,
            "generations": 200,
            "num_agents": 250,
            "top_limit": 5,
            "n_survivors": 1,
            "using_biases": True,
            "layer": layer,
            "use_recomb": recomb,
            "recomb_meth": recomb_meth,
            "seed": seed,
        },
    )
    config: Config = wandb.config

    # Seeding Method
    def initialize_rng(seed_value: int):
        """seed everything"""
        env = runner.create_env()
        env.reset(seed_value)
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.use_deterministic_algorithms(True)

        return np.random.default_rng(seed_value)

    # Seed everything and create env
    initialize_rng(seed_value=config.seed)
    env = runner.create_env()

    # initialize random agents
    agents = runner.initialize_random_agents(config.num_agents, env, layer)

    # final parameters
    averages = []
    std_deviations = []
    averages_top = []
    std_deviations_top = []

    # parameters for dataframe
    mutation_power_d = []
    generations_d = []
    num_agents_d = []
    top_limit_d = []
    n_survivors_d = []
    using_biases_d = []
    seed_d = []
    current_generation_d = []
    avg_score_d = []
    std_dev_d = []
    test_d = []
    own_coin_avg_d = []
    coin_avg_d = []
    time_d = []

    # for every generation
    for generation in range(config.generations):
        start_time = time.time()
        # reset scores
        scores = [] * config.num_agents
        coin_total = [] * config.num_agents
        own_coin_total = [] * config.num_agents

        # for every agent
        for i, _ in enumerate(agents):
            agent_1 = agents[i]
            agent_2 = agents[i]

            # run the agent averaging over n times
            score_dict, coin_total_dict, own_coin_total_dict = runner.run_multi_agent(
                env, {"player_1": agent_1, "player_2": agent_2}, seed=config.seed
            )

            # append all data collected from run
            scores.append(score_dict["player_1"] + score_dict["player_2"])
            coin_total.append(coin_total_dict["player_1"] + coin_total_dict["player_2"])
            own_coin_total.append(
                own_coin_total_dict["player_1"] + own_coin_total_dict["player_2"]
            )

        # calculate Averages
        score_avg = sum(scores, start=0) / len(scores)
        coin_avg = sum(coin_total, start=0) / len(coin_total)
        own_coin_avg = sum(own_coin_total, start=0) / len(own_coin_total)

        # depending on the scores, sort the indices
        sorted_indices = np.argsort(scores)

        # take the indices of the last top n agents
        top_indices = sorted_indices[-config.top_limit :]

        # type: ignore
        # reproduce agents with either recombination or mutation or both

        # Mutation:
        # children = runner.reproduce_remcomb(agents, list(top_indices), scores)

        # Mixed:
        if recomb:
            # "Recombine and Mutate"
            children_v = runner.reproduce_remcomb(
                agents, list(top_indices), scores, layer, recomb_meth
            )
            children = runner.reproduce(
                children_v, list(top_indices), scores, config.mutation_power
            )
        else:
            # "Mutation only"
            children = runner.reproduce(agents, list(top_indices), scores, config.mutation_power)

        # replace generation
        agents = children

        # for simplicity, convert to numpy array
        top_indices = np.array(top_indices)
        scores = np.array(scores)

        print(scores[top_indices])
        print("")

        # save avg
        avg = np.mean(scores)
        averages.append(avg)

        avg_top = np.mean(scores[top_indices])
        averages_top.append(avg_top)

        # save std_dev
        std_dev = np.std(scores)
        std_deviations.append(std_dev)

        wandb.log({"std_deviations": std_dev}, step=generation)

        std_dev_top = np.std(scores[top_indices])
        std_deviations_top.append(std_dev_top)

        # print circuit
        # agents[0].show(torch.tensor([0, 0, 0, 0]))

        mutation_power_d.append(config.mutation_power)
        generations_d.append(config.generations)
        num_agents_d.append(config.num_agents)
        top_limit_d.append(config.top_limit)
        n_survivors_d.append(config.n_survivors)
        using_biases_d.append(config.using_biases)
        seed_d.append(config.seed)
        current_generation_d.append(generation + 1)
        avg_score_d.append(score_avg)
        std_dev_d.append(std_dev)
        test_d.append(f"VQC-{layer}-{evo_meth}")
        own_coin_avg_d.append(own_coin_avg)
        coin_avg_d.append(coin_avg)
        time_d.append(time.time() - start_time)

        # print info
        print(
            " Generation ",
            generation,
            " | Mean rewards: ",
            avg,
            "| Std dev:  ",
            std_dev,
            "| Mean of top ",
            config.top_limit,
            ": ",
            avg_top,
            "Std dev elite: ",
            std_dev_top,
        )
        # create dataframe
        data = {
            "mutation_power": mutation_power_d,
            "num_generations": generations_d,
            "num_agents": num_agents_d,
            "top_limit": top_limit_d,
            "n_survivors": n_survivors_d,
            "using_biases": using_biases_d,
            "seed": seed_d,
            "current_generation": current_generation_d,
            "avg_score": avg_score_d,
            "std_dev": std_dev_d,
            "Tests": test_d,
            "own_coin_avg": own_coin_avg_d,
            "coin_avg": coin_avg_d,
            "Time": time_d,
        }
        df = pd.DataFrame(data)
        print(df)
        df.to_csv(f"Data-{layer}-{evo_meth}-{seed}.csv")


if __name__ == "__main__":
    s, l, r, m = parse_args()

    main(s, l, r, m)
