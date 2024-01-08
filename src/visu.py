"""Visualize data from data.csv"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import rc

# Dataframes of all Data collected during the Experiments
data_NN21 = pd.concat(
    map(
        pd.read_csv,
        [
            "C:\\Users\\user\\Desktop\\BA-Daten\\NN21\\DataNN2x10.csv",
            "C:\\Users\\user\\Desktop\\BA-Daten\\NN21\\DataNN2x11.csv",
            "C:\\Users\\user\\Desktop\\BA-Daten\\NN21\\DataNN2x12.csv",
            "C:\\Users\\user\\Desktop\\BA-Daten\\NN21\\DataNN2x13.csv",
            "C:\\Users\\user\\Desktop\\BA-Daten\\NN21\\DataNN2x14.csv",
        ],
    ),
    ignore_index=True,
)

data_NN64_time = pd.concat(
    map(
        pd.read_csv,
        [
            "C:\\Users\\user\\Desktop\\BA-Daten\\NN64_Time\\DataNN640.csv",
            "C:\\Users\\user\\Desktop\\BA-Daten\\NN64_Time\\DataNN641.csv",
            "C:\\Users\\user\\Desktop\\BA-Daten\\NN64_Time\\DataNN642.csv",
            "C:\\Users\\user\\Desktop\\BA-Daten\\NN64_Time\\DataNN643.csv",
            "C:\\Users\\user\\Desktop\\BA-Daten\\NN64_Time\\DataNN644.csv",
        ],
    ),
    ignore_index=True,
)


data_Muta = pd.concat(
    map(
        pd.read_csv,
        [
            "C:\\Users\\user\\Desktop\\BA-Daten\\Muta\\DataMuta0.csv",
            "C:\\Users\\user\\Desktop\\BA-Daten\\Muta\\DataMuta1.csv",
            "C:\\Users\\user\\Desktop\\BA-Daten\\Muta\\DataMuta2.csv",
            "C:\\Users\\user\\Desktop\\BA-Daten\\Muta\\DataMuta3.csv",
            "C:\\Users\\user\\Desktop\\BA-Daten\\Muta\\DataMuta4.csv",
        ],
    ),
    ignore_index=True,
)
print(data_Muta)

# Data with RecombMuta and variable Layer number
# 2 Layer with time
data_2Layer = pd.concat(
    map(
        pd.read_csv,
        [
            "C:\\Users\\user\\Desktop\\BA-Daten\\2Layer\\DataVQC20.csv",
            "C:\\Users\\user\\Desktop\\BA-Daten\\2Layer\\DataVQC21.csv",
            "C:\\Users\\user\\Desktop\\BA-Daten\\2Layer\\DataVQC22.csv",
            "C:\\Users\\user\\Desktop\\BA-Daten\\2Layer\\DataVQC23.csv",
            "C:\\Users\\user\\Desktop\\BA-Daten\\2Layer\\DataVQC24.csv",
        ],
    ),
    ignore_index=True,
)


# 4 Layer
data_RecombMuta = pd.concat(
    map(
        pd.read_csv,
        [
            "C:\\Users\\user\\Desktop\\BA-Daten\\RecombMuta\\DataRecombMuta0.csv",
            "C:\\Users\\user\\Desktop\\BA-Daten\\RecombMuta\\DataRecombMuta1.csv",
            "C:\\Users\\user\\Desktop\\BA-Daten\\RecombMuta\\DataRecombMuta2.csv",
            "C:\\Users\\user\\Desktop\\BA-Daten\\RecombMuta\\DataRecombMuta3.csv",
            "C:\\Users\\user\\Desktop\\BA-Daten\\RecombMuta\\DataRecombMuta4.csv",
        ],
    ),
    ignore_index=True,
)

data_4Layer = pd.concat(
    map(
        pd.read_csv,
        [
            "C:\\Users\\user\\Desktop\\BA-Daten\\4Layer_Time\\DataVQC40.csv",
            "C:\\Users\\user\\Desktop\\BA-Daten\\4Layer_Time\\DataVQC41.csv",
            "C:\\Users\\user\\Desktop\\BA-Daten\\4Layer_Time\\DataVQC42.csv",
            "C:\\Users\\user\\Desktop\\BA-Daten\\4Layer_Time\\DataVQC43.csv",
            "C:\\Users\\user\\Desktop\\BA-Daten\\4Layer_Time\\DataVQC44.csv",
        ],
    ),
    ignore_index=True,
)

# with time
data_6Layer_time = pd.concat(
    map(
        pd.read_csv,
        [
            "C:\\Users\\user\\Desktop\\BA-Daten\\6Layer_Time\\DataVQC60.csv",
            "C:\\Users\\user\\Desktop\\BA-Daten\\6Layer_Time\\DataVQC61.csv",
            "C:\\Users\\user\\Desktop\\BA-Daten\\6Layer_Time\\DataVQC62.csv",
            "C:\\Users\\user\\Desktop\\BA-Daten\\6Layer_Time\\DataVQC63.csv",
            "C:\\Users\\user\\Desktop\\BA-Daten\\6Layer_Time\\DataVQC64.csv",
        ],
    ),
    ignore_index=True,
)

# with time
data_8Layer = pd.concat(
    map(
        pd.read_csv,
        [
            "C:\\Users\\user\\Desktop\\BA-Daten\\8Layer\\DataVQC80.csv",
            "C:\\Users\\user\\Desktop\\BA-Daten\\8Layer\\DataVQC81.csv",
            "C:\\Users\\user\\Desktop\\BA-Daten\\8Layer\\DataVQC82.csv",
            "C:\\Users\\user\\Desktop\\BA-Daten\\8Layer\\DataVQC83.csv",
            "C:\\Users\\user\\Desktop\\BA-Daten\\8Layer\\DataVQC84.csv",
        ],
    ),
    ignore_index=True,
)


data_Random = pd.concat(
    map(
        pd.read_csv,
        [
            "C:\\Users\\user\\Desktop\\BA-Daten\\Random\\DataRandom0.csv",
            "C:\\Users\\user\\Desktop\\BA-Daten\\Random\\DataRandom1.csv",
            "C:\\Users\\user\\Desktop\\BA-Daten\\Random\\DataRandom2.csv",
            "C:\\Users\\user\\Desktop\\BA-Daten\\Random\\DataRandom3.csv",
            "C:\\Users\\user\\Desktop\\BA-Daten\\Random\\DataRandom4.csv",
        ],
    ),
    ignore_index=True,
)


data_Muta_005 = pd.concat(
    map(
        pd.read_csv,
        [
            "C:\\Users\\user\\Desktop\\BA-Daten\\Muta_0.05\\DataVQC_Muta_050.csv",
            "C:\\Users\\user\\Desktop\\BA-Daten\\Muta_0.05\\DataVQC_Muta_051.csv",
            "C:\\Users\\user\\Desktop\\BA-Daten\\Muta_0.05\\DataVQC_Muta_052.csv",
            "C:\\Users\\user\\Desktop\\BA-Daten\\Muta_0.05\\DataVQC_Muta_053.csv",
            "C:\\Users\\user\\Desktop\\BA-Daten\\Muta_0.05\\DataVQC_Muta_054.csv",
        ],
    ),
    ignore_index=True,
)

data_Muta_02 = pd.concat(
    map(
        pd.read_csv,
        [
            "C:\\Users\\user\\Desktop\\BA-Daten\\Muta_0.02\\DataVQC_Muta0.csv",
            "C:\\Users\\user\\Desktop\\BA-Daten\\Muta_0.02\\DataVQC_Muta1.csv",
            "C:\\Users\\user\\Desktop\\BA-Daten\\Muta_0.02\\DataVQC_Muta2.csv",
            "C:\\Users\\user\\Desktop\\BA-Daten\\Muta_0.02\\DataVQC_Muta3.csv",
            "C:\\Users\\user\\Desktop\\BA-Daten\\Muta_0.02\\DataVQC_Muta4.csv",
        ],
    ),
    ignore_index=True,
)

# New Paper Grafics:
data_6Layer_LayerwiseRecomb = pd.concat(
    map(
        pd.read_csv,
        [
            "C:\\Users\\user\\Desktop\\Paper-Daten\\6L-200G-LayerwiseRecomb\\DataVQC6DR0.csv",
            "C:\\Users\\user\\Desktop\\Paper-Daten\\6L-200G-LayerwiseRecomb\\DataVQC6DR1.csv",
            "C:\\Users\\user\\Desktop\\Paper-Daten\\6L-200G-LayerwiseRecomb\\DataVQC6DR2.csv",
            "C:\\Users\\user\\Desktop\\Paper-Daten\\6L-200G-LayerwiseRecomb\\DataVQC6DR3.csv",
            "C:\\Users\\user\\Desktop\\Paper-Daten\\6L-200G-LayerwiseRecomb\\DataVQC6DR4.csv",
        ],
    ),
    ignore_index=True,
)

data_6Layer_randomRecomb = pd.concat(
    map(
        pd.read_csv,
        [
            "C:\\Users\\user\\Desktop\\Paper-Daten\\6L-200G-RandomRecomb\\DataVQC60.csv",
            "C:\\Users\\user\\Desktop\\Paper-Daten\\6L-200G-RandomRecomb\\DataVQC61.csv",
            "C:\\Users\\user\\Desktop\\Paper-Daten\\6L-200G-RandomRecomb\\DataVQC62.csv",
            "C:\\Users\\user\\Desktop\\Paper-Daten\\6L-200G-RandomRecomb\\DataVQC63.csv",
            "C:\\Users\\user\\Desktop\\Paper-Daten\\6L-200G-RandomRecomb\\DataVQC64.csv",
        ],
    ),
    ignore_index=True,
)

# Fixing Name of Data
data_6Layer_LayerwiseRecomb = data_6Layer_LayerwiseRecomb.replace(
    ["VQC(6): ReMu"],
    ["VQC(6): LaRe"],
)

data_6Layer_RandomRecomb = data_6Layer_RandomRecomb.replace(
    ["VQC(6): ReMu"],
    ["VQC(6): LaRe"],
)

# Smooth the lines
data_Muta["avg_score"] = data_Muta["avg_score"].ewm(span=5).mean()
data_2Layer["avg_score"] = data_2Layer["avg_score"].ewm(span=5).mean()
data_4Layer["avg_score"] = data_4Layer["avg_score"].ewm(span=5).mean()
data_6Layer_time["avg_score"] = data_6Layer_time["avg_score"].ewm(span=5).mean()
data_8Layer["avg_score"] = data_8Layer["avg_score"].ewm(span=5).mean()
data_NN21["avg_score"] = data_NN21["avg_score"].ewm(span=5).mean()
data_NN64_time["avg_score"] = data_NN64_time["avg_score"].ewm(span=5).mean()
data_6Layer_randomRecomb["avg_score"] = data_6Layer_randomRecomb["avg_score"].ewm(span=5).mean()
data_6Layer_LayerwiseRecomb["avg_score"] = (
    data_6Layer_LayerwiseRecomb["avg_score"].ewm(span=5).mean()
)


data_Muta["coin_avg"] = data_Muta["coin_avg"].ewm(span=5).mean()
data_2Layer["coin_avg"] = data_2Layer["coin_avg"].ewm(span=5).mean()
data_4Layer["coin_avg"] = data_4Layer["coin_avg"].ewm(span=5).mean()
data_6Layer_time["coin_avg"] = data_6Layer_time["coin_avg"].ewm(span=5).mean()
data_8Layer["coin_avg"] = data_8Layer["coin_avg"].ewm(span=5).mean()
data_NN21["coin_avg"] = data_NN21["coin_avg"].ewm(span=5).mean()
data_NN64_time["coin_avg"] = data_NN64_time["coin_avg"].ewm(span=5).mean()
data_6Layer_randomRecomb["coin_avg"] = data_6Layer_randomRecomb["coin_avg"].ewm(span=5).mean()
data_6Layer_LayerwiseRecomb["coin_avg"] = (
    data_6Layer_LayerwiseRecomb["coin_avg"].ewm(span=5).mean()
)


data_Muta["own_coin_avg"] = data_Muta["own_coin_avg"].ewm(span=5).mean()
data_2Layer["own_coin_avg"] = data_2Layer["own_coin_avg"].ewm(span=5).mean()
data_4Layer["own_coin_avg"] = data_4Layer["own_coin_avg"].ewm(span=5).mean()
data_6Layer_time["own_coin_avg"] = data_6Layer_time["own_coin_avg"].ewm(span=5).mean()
data_8Layer["own_coin_avg"] = data_8Layer["own_coin_avg"].ewm(span=5).mean()
data_NN21["own_coin_avg"] = data_NN21["own_coin_avg"].ewm(span=5).mean()
data_NN64_time["own_coin_avg"] = data_NN64_time["own_coin_avg"].ewm(span=5).mean()
data_6Layer_randomRecomb["own_coin_avg"] = (
    data_6Layer_randomRecomb["own_coin_avg"].ewm(span=5).mean()
)
data_6Layer_LayerwiseRecomb["own_coin_avg"] = (
    data_6Layer_LayerwiseRecomb["own_coin_avg"].ewm(span=5).mean()
)


# create Datafram for VQC vs. Random and vs. NN
data_BA1 = pd.concat(
    [
        data_Muta,
        data_4Layer,
        data_NN21,
        data_NN64_time,
        data_Random,
    ],
    ignore_index=True,
)

# create Dataframe for VQCs with different number of Layers
data_BA2 = pd.concat(
    [data_NN64_time, data_2Layer, data_4Layer, data_6Layer_time, data_8Layer],
    ignore_index=True,
)

# Paper Dataframe
data_Paper1 = pd.concat(
    [data_6Layer_randomRecomb, data_6Layer_LayerwiseRecomb],
    ignore_index=True,
)

# calculate own_coin_rate and Rename some Data for paper
data_Paper1["own_coin_rate"] = data_Paper1["own_coin_avg"] / data_Paper1["coin_avg"]
data_Paper1.rename(columns={"coin_avg": "Average collected coins"}, inplace=True)
data_Paper1.rename(columns={"avg_score": "Average score"}, inplace=True)
data_Paper1.rename(columns={"own_coin_avg": "Average collected own coins"}, inplace=True)
data_Paper1.rename(columns={"current_generation": "Generation"}, inplace=True)
data_Paper1.rename(columns={"Tests": "Experiments"}, inplace=True)
data_Paper1.rename(columns={"own_coin_rate": "Own coin rate"}, inplace=True)
data_Paper1.rename(columns={"Time": "Ausführungszeit in Sekunden"}, inplace=True)

# calculate own-coin-rate and Rename some Data
data_BA1["own_coin_rate"] = data_BA1["own_coin_avg"] / data_BA1["coin_avg"]
data_BA1.rename(columns={"coin_avg": "Average collected coins"}, inplace=True)
data_BA1.rename(columns={"avg_score": "Average score"}, inplace=True)
data_BA1.rename(columns={"own_coin_avg": "Average collected own coins"}, inplace=True)
data_BA1.rename(columns={"current_generation": "Generation"}, inplace=True)
data_BA1.rename(columns={"Tests": "Experiments"}, inplace=True)
data_BA1.rename(columns={"own_coin_rate": "Own coin rate"}, inplace=True)
data_BA1 = data_BA1.replace(
    ["Muta: 0.01", "RecombMuta", "NN2x1", "NN64x64: ReMu"],
    ["VQC(4): Muta", "VQC: ReMu", "NN(2x1): ReMu", "NN(64x64): ReMu"],
)

# calculate own-coin-rate and Rename some Data
data_BA2["own_coin_rate"] = data_BA2["own_coin_avg"] / data_BA2["coin_avg"]
data_BA2.rename(columns={"coin_avg": "Average collected coins"}, inplace=True)
data_BA2.rename(columns={"avg_score": "Average score"}, inplace=True)
data_BA2.rename(columns={"own_coin_avg": "Average collected own coins"}, inplace=True)
data_BA2.rename(columns={"current_generation": "Generation"}, inplace=True)
data_BA2.rename(columns={"Tests": "Experiments"}, inplace=True)
data_BA2.rename(columns={"own_coin_rate": "Own coin rate"}, inplace=True)
data_BA2.rename(columns={"Time": "Ausführungszeit in Sekunden"}, inplace=True)
data_BA2 = data_BA2.replace(
    ["VQC(2): ReMu", "VQC(4): ReMu", "VQC(6): ReMu", "VQC(8): ReMu", "NN64x64: ReMu"],
    ["VQC(2)", "VQC(4)", "VQC(6)", "VQC(8)", "NN(64x64)"],
)

# Create Plots
sns.set(font_scale=1.4)
sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))
rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
rc("text", usetex=True)
fig = sns.lineplot(  # lineplot für alles was nicht Zeitplot ist, barplot für Zeitplot
    data=data_Paper1,
    # x="Experiments",                  # Timeplot
    # y="Ausführungszeit in Sekunden",  # Timeplot
    x="Generation",  # Resultsplot
    # y="Average collected coins",
    y="Average score",
    # y="Average collected own coins",
    # y="Own coin rate",
    hue="Experiments",
    palette="colorblind",
)

plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig("PaperScore.pdf")
