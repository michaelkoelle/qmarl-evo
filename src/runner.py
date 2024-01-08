"""Runner for main trainings loop"""

import copy
import random
from math import ceil, log2
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from gym.spaces import Discrete
from pettingzoo.utils import AECEnv
from torch import nn

from env import CoinGame

# from neuralnetwork import NeuralNetwork
from vqc import VQC


def create_env():
    """Creats the environment"""
    env = CoinGame(3, 3)
    return env


def initialize_random_agents(num_agents: int, env: AECEnv, layer: int) -> List[nn.Module]:
    """
    Returns the first randomly generated agents
    """
    agents = []

    obsspace: Dict[str, np.ndarray[float, Any]] = env.observation_space("player_1")  # type: ignore
    obs = obsspace.get("observation")
    if obs is None:
        raise Exception("Fehler: Key Observation gibt es nicht im Observationspace")
    observationspace = obs.shape[0] * obs.shape[1] * obs.shape[2]
    actionspace: Discrete = env.action_space("player_1")  # type: ignore
    num_qubits = max(actionspace.n, ceil(log2(observationspace)))
    for _ in range(num_agents):
        # create the agent either VQC or NN
        # VQC:
        agent = VQC(num_qubits, layer, actionspace.n)

        # NN:
        # agent = NeuralNetwork(observationspace, actionspace.n)
        # for param in agent.parameters():
        # param.requires_grad = False

        # num_params = sum([np.prod(p.size()) for p in agent.parameters()])  # type: ignore
        # print(num_params)

        # add the agent
        agents.append(agent)

    return agents


def policy(observation: Dict[str, np.ndarray[float, Any]], agent: nn.Module):
    """Define policy"""

    def map_to_intervall(
        x: Any, input_min: float, input_max: float, output_min: float, output_max: float
    ):
        return (x - input_min) * (output_max - output_min) / (input_max - input_min) + output_min

    obs = observation.get("observation")
    if obs is None:
        raise Exception("Fehler: Key Observation gibt es nicht im Observationspace")
    action_mask = observation.get("action_mask")
    if action_mask is None:
        raise Exception("Fehler: Key Observation gibt es nicht im Observationspace")
    action_values = agent(torch.tensor(np.array([obs.flatten()])))
    act = map_to_intervall(action_values, -1, 1, 0, 1) * torch.tensor(action_mask)
    action = torch.argmax(act)

    return action.item()


def run_multi_agent(
    env: AECEnv, agents: Dict[str, nn.Module], seed: int
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    For every agent, run a whole episode n times
    """
    # reset environment and create empty scores list for the run
    scores = {"player_1": 0.0, "player_2": 0.0}
    coin_total = {"player_1": 0.0, "player_2": 0.0}
    other_coin_total = {"player_1": 0.0, "player_2": 0.0}
    own_coin_total = {"player_1": 0.0, "player_2": 0.0}
    env.reset(seed)
    for a in env.agent_iter():
        agent = agents.get(a)
        if agent is None:
            raise Exception(f"Es gibt keinen Agent für {a}")
        observation, reward, done, _ = env.last()
        scores[a] = reward + scores.get(a, 0)
        if reward == -2:
            other_coin_total["player_1" if a == "player_2" else "player_2"] += 1
        elif reward == +1:
            coin_total[a] += 1

        if observation is None:
            raise Exception("Fehler: Observation darf nicht null sein")

        # for action taken with policy
        action = policy(observation, agent) if not done else None

        # for random action
        # action_mask = observation.get("action_mask")

        # def check_action(x: int) -> bool:
        # return action_mask[x]

        # valid_action = list(filter(check_action, range(4)))
        # action = random.choice(valid_action) if not done else None

        # Render Env and take Action in Env
        # env.render()
        env.step(action)

        # env.render()

        # calculate total own coins
        own_coin_total = {k: coin_total[k] - v for (k, v) in other_coin_total.items()}
    return scores, coin_total, own_coin_total


def mutate(agent: nn.Module, mutation_power: float):
    """
    Function that determines the mutation of future generations
    """
    # copy agent
    child_agent = copy.deepcopy(agent)

    # for every parameter
    for param in child_agent.parameters():
        # save param shape
        shape = param.shape

        # reshape tensor into one line
        linear = torch.reshape(param, (-1,))

        # mutate values
        for i, _ in enumerate(linear):
            linear[i] += mutation_power * np.random.randn()

        # reshape tensor to old shape
        param = torch.reshape(linear, shape)

    return child_agent


def recombinate(agent_m: nn.Module, agent_f: nn.Module, layer: int, method: str):
    """Function that creates new agents for future generations through recombination"""
    mother_child = copy.deepcopy(agent_m)
    father_child = copy.deepcopy(agent_f)

    num_model_params = sum(int(np.prod(p.size())) for p in mother_child.parameters())

    if method == "layerwise":
        # layerwise crossover
        num_layer_params = 18
        l = np.random.randint(0, layer)
        z = l * num_layer_params
    else:
        # random point of crossover
        z = np.random.randint(0, num_model_params)

    mother_params = flatten_params(mother_child)
    father_params = flatten_params(father_child)

    # single-point crossover
    recombined_params = torch.cat([mother_params[:z], father_params[z:]])

    assign_params(mother_child, recombined_params)

    return mother_child


def flatten_params(model: nn.Module):
    """Flatten model parameters into a single vector."""
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def assign_params(model: nn.Module, params: torch.Tensor):
    """Assign flattened parameters to the model."""
    param_idx = 0
    for p in model.parameters():
        num_params = np.prod(p.size())
        p.data = params[param_idx : param_idx + num_params].view(p.size())
        param_idx += num_params


def reproduce(
    agents: List[nn.Module],
    top_indices: List[int],
    scores: List[float],
    mutation_power: float,
):
    """
    Reproduce top agents
    """
    children = []

    # N-1 agents will mutate and die
    for _ in range(len(agents) - 1):
        # select random index
        random_index = random.choice(top_indices)

        # mutate selected agent
        mutated_agent = mutate(agents[random_index], mutation_power)

        # append mutated agent to children
        children.append(mutated_agent)

    # 1 agents will survive
    # select best agent
    best_agent = np.argmax(scores)

    # this agent will survive
    children.append(agents[best_agent])

    return children


def reproduce_remcomb(
    agents: List[nn.Module],
    top_indices: List[int],
    scores: List[float],
    layer: int, 
    method: str
):
    """
    Reproduce top agents
    """
    children = []

    # N-1 agents will mutate and die
    for _ in range(len(agents) - 1):
        # Select parents
        random_index = random.choice(top_indices)
        random_index2 = random.choice(top_indices)

        random_index2_pos = top_indices.index(random_index2)

        if random_index == random_index2:
            if random_index2_pos + 1 < len(top_indices):
                random_index2 = top_indices[random_index2_pos + 1]

            elif random_index2_pos - 1 > 0:
                random_index2 = top_indices[random_index2_pos - 1]

        mutated_agent = recombinate(agents[random_index], agents[random_index2], layer, method)

        # append mutated agent to children
        children.append(mutated_agent)

    # 1 agents will survive
    # select best agent
    best_agent = np.argmax(scores)

    # this agent will survive
    children.append(agents[best_agent])

    return children
