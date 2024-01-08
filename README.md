
# Multi-Agent Quantum Reinforcement Learning using Evolutionary Optimization

## Abstract

Multi-Agent Reinforcement Learning is pivotal in the era of autonomous driving and smart industrial applications. This project takes a novel approach by integrating quantum mechanics into Reinforcement Learning, significantly reducing the model's trainable parameters. Traditional gradient-based methods in Multi-Agent Quantum Reinforcement Learning face challenges like barren plateaus, limiting their effectiveness compared to classical approaches. Our work builds upon an existing gradient-free Quantum Reinforcement Learning method and introduces three variants using Variational Quantum Circuits for Multi-Agent Reinforcement Learning, employing evolutionary optimization. We test these variants in the Coin Game environment, comparing them with classical models. Our findings indicate that Variational Quantum Circuit methods have a distinct advantage over comparable neural networks in terms of trainable parameters, achieving similar results with 97.88% fewer parameters.

## Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

The `src/main.py` script evaluates the proposed Quantum Reinforcement Learning models. It requires the following arguments:

* `-s` or `--seed`: Seed value (integer).
* `-l` or `--layer`: Layer value (integer).
* `-e` or `--experiment`: Experiment type (string).

To run the script:

```
python src/main.py --seed [SEED_VALUE] --layer [LAYER_VALUE] --experiment [EXPERIMENT_TYPE]
```

Replace `[SEED_VALUE]`, `[LAYER_VALUE]`, and `[EXPERIMENT_TYPE]` with your chosen values.

Experiment types:

* `muta`: For mutation experiments.
* `recomb_random`: For random recombination experiments.
* `recomb_layerwise`: For layer-wise recombination experiments.
