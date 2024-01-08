"""Visualize data from data.csv"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# import seaborn as sns

# datas = pd.read_csv("C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4_Layer\\Data1.csv")

# Seed 0 -9 with 200 generations mutationpower = 0.02 and 4 Layers
datas = pd.concat(
    map(
        pd.read_csv,
        [
            "C:\\Users\\felix\\Desktop\\BA-Daten\\Test_Muta_0.02\\DataMuta_0.020.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten\\Test_Muta_0.02\\DataMuta_0.021.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten\\Test_Muta_0.02\\DataMuta_0.022.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten\\Test_Muta_0.02\\DataMuta_0.023.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten\\Test_Muta_0.02\\DataMuta_0.024.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten\\Test_Muta_0.02\\DataMuta_0.025.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten\\Test_Muta_0.02\\DataMuta_0.026.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten\\Test_Muta_0.02\\DataMuta_0.027.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten\\Test_Muta_0.02\\DataMuta_0.028.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten\\Test_Muta_0.02\\DataMuta_0.029.csv",
        ],
    ),
    ignore_index=True,
)

# Seed 0 -9 with 200 generations mutationpower = 0.02, 4 Layers, corrected Env
datas_v2 = pd.concat(
    map(
        pd.read_csv,
        [
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2\\Data0.020.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2\\Data0.021.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2\\Data0.022.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2\\Data0.023.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2\\Data0.024.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2\\Data0.025.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2\\Data0.026.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2\\Data0.027.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2\\Data0.028.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2\\Data0.029.csv",
        ],
    ),
    ignore_index=True,
)

# Seed 0 -9 with 200 generations mutationpower = 0.01 and 4 Layers
datas_2 = pd.concat(
    map(
        pd.read_csv,
        [
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4_Layer_0.01_10\\Data0.010.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4_Layer_0.01_10\\Data0.011.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4_Layer_0.01_10\\Data0.012.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4_Layer_0.01_10\\Data0.013.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4_Layer_0.01_10\\Data0.014.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4_Layer_0.01_10\\Data0.015.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4_Layer_0.01_10\\Data0.016.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4_Layer_0.01_10\\Data0.017.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4_Layer_0.01_10\\Data0.018.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4_Layer_0.01_10\\Data0.019.csv",
        ],
    ),
    ignore_index=True,
)

# Seed 0 - 9 with 200 generations mutationpower = 0.01 and 4 Layers, corrected Env
datas2_v2 = pd.concat(
    map(
        pd.read_csv,
        [
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_0.01\\Data0.010.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_0.01\\Data0.011.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_0.01\\Data0.012.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_0.01\\Data0.013.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_0.01\\Data0.014.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_0.01\\Data0.015.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_0.01\\Data0.016.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_0.01\\Data0.017.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_0.01\\Data0.018.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_0.01\\Data0.019.csv",
        ],
    ),
    ignore_index=True,
)

# Seed 0 - 9 with 200 generations mutationpower = 0.03 and 4 Layers
datas_3 = pd.concat(
    map(
        pd.read_csv,
        [
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4_Layer_0.03_10\\Data0.030.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4_Layer_0.03_10\\Data0.031.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4_Layer_0.03_10\\Data0.032.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4_Layer_0.03_10\\Data0.033.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4_Layer_0.03_10\\Data0.034.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4_Layer_0.03_10\\Data0.035.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4_Layer_0.03_10\\Data0.036.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4_Layer_0.03_10\\Data0.037.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4_Layer_0.03_10\\Data0.038.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4_Layer_0.03_10\\Data0.039.csv",
        ],
    ),
    ignore_index=True,
)

# Seed 0 - 9 with 200 generations mutationpower = 0.03 and 4 Layers, corrected Env
datas3_v2 = pd.concat(
    map(
        pd.read_csv,
        [
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_0.03\\Data0.030.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_0.03\\Data0.031.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_0.03\\Data0.032.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_0.03\\Data0.033.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_0.03\\Data0.034.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_0.03\\Data0.035.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_0.03\\Data0.036.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_0.03\\Data0.037.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_0.03\\Data0.038.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_0.03\\Data0.039.csv",
        ],
    ),
    ignore_index=True,
)

# Seed 0 - 5 with 200 generations mutationpower = 0.02 and 6 Layers
datas_4 = pd.concat(
    map(
        pd.read_csv,
        [
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_6_Layer\\DataTest_2_6_Layers0.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_6_Layer\\DataTest_2_6_Layers1.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_6_Layer\\DataTest_2_6_Layers2.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_6_Layer\\DataTest_2_6_Layers3.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_6_Layer\\DataTest_2_6_Layers4.csv",
        ],
    ),
    ignore_index=True,
)

# Seed 0 - 9 with 200 generations mutationpower = 0.02 and 6 Layers, corrected Env
datas4_v2 = pd.concat(
    map(
        pd.read_csv,
        [
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_6Layer_v2\\Data6_Layer0.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_6Layer_v2\\Data6_Layer1.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_6Layer_v2\\Data6_Layer2.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_6Layer_v2\\Data6_Layer3.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_6Layer_v2\\Data6_Layer4.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_6Layer_v2\\Data_2_6_Layers5.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_6Layer_v2\\Data_2_6_Layers6.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_6Layer_v2\\Data_2_6_Layers7.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_6Layer_v2\\Data_2_6_Layers8.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_6Layer_v2\\Data_2_6_Layers9.csv",
        ],
    ),
    ignore_index=True,
)

# Seeds 0 - 5 with 300 generations mutationpower = 0.02 and 4 Layers
datas_5 = pd.concat(
    map(
        pd.read_csv,
        [
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4_Layer_300_gen\\Data3000.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4_Layer_300_gen\\Data3001.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4_Layer_300_gen\\Data3002.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4_Layer_300_gen\\Data3003.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4_Layer_300_gen\\Data3004.csv",
        ],
    ),
    ignore_index=True,
)

# Seeds 0 - 9 recomb with 200 generations and 4 Layers
datas_6 = pd.concat(
    map(
        pd.read_csv,
        [
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_recomb\\Datarecomb0.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_recomb\\Datarecomb1.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_recomb\\Datarecomb2.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_recomb\\Datarecomb3.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_recomb\\Datarecomb4.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_recomb\\Datarecomb5.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_recomb\\Datarecomb6.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_recomb\\Datarecomb7.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_recomb\\Datarecomb8.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_recomb\\Datarecomb9.csv",
        ],
    ),
    ignore_index=True,
)

# Seeds 0 - 9 recomb with 200 generations and 4 Layers, corrected Env
datas6_v2 = pd.concat(
    map(
        pd.read_csv,
        [
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_recomb\\Datarecomb0.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_recomb\\Datarecomb1.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_recomb\\Datarecomb2.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_recomb\\Datarecomb3.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_recomb\\Datarecomb4.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_recomb\\Datarecomb5.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_recomb\\Datarecomb6.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_recomb\\Datarecomb7.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_recomb\\Datarecomb8.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_recomb\\Datarecomb9.csv",
        ],
    ),
    ignore_index=True,
)

# Seeds 0 - 9 recomb + mutation with 200 generations and 4 Layers, corrected Env
datas7_v2 = pd.concat(
    map(
        pd.read_csv,
        [
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_recombmuta\\Datarecomb_mut0.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_recombmuta\\Datarecomb_mut1.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_recombmuta\\Datarecomb_mut2.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_recombmuta\\Datarecomb_mut3.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_recombmuta\\Datarecomb_mut4.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_recombmuta\\Datarecomb_mut5.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_recombmuta\\Datarecomb_mut6.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_recombmuta\\Datarecomb_mut7.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_recombmuta\\Datarecomb_mut8.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_recombmuta\\Datarecomb_mut9.csv",
        ],
    ),
    ignore_index=True,
)

# All Datas combined
datas_8 = pd.concat(
    map(
        pd.read_csv,
        [
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4_Layer\\Data0.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4_Layer\\Data1.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4_Layer\\Data2.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4_Layer\\Data3.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4_Layer\\Data4.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4_Layer\\Data5.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4_Layer\\Data6.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4_Layer\\Data7.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4_Layer\\Data8.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4_Layer\\Data9.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4_Layer_0.01_10\\Data0.010.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4_Layer_0.01_10\\Data0.011.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4_Layer_0.01_10\\Data0.012.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4_Layer_0.01_10\\Data0.013.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4_Layer_0.01_10\\Data0.014.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4_Layer_0.01_10\\Data0.015.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4_Layer_0.01_10\\Data0.016.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4_Layer_0.01_10\\Data0.017.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4_Layer_0.01_10\\Data0.018.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4_Layer_0.01_10\\Data0.019.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4_Layer_0.03_10\\Data0.030.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4_Layer_0.03_10\\Data0.031.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4_Layer_0.03_10\\Data0.032.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4_Layer_0.03_10\\Data0.033.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4_Layer_0.03_10\\Data0.034.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4_Layer_0.03_10\\Data0.035.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4_Layer_0.03_10\\Data0.036.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4_Layer_0.03_10\\Data0.037.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4_Layer_0.03_10\\Data0.038.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4_Layer_0.03_10\\Data0.039.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_recomb\\Datarecomb0.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_recomb\\Datarecomb1.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_recomb\\Datarecomb2.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_recomb\\Datarecomb3.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_recomb\\Datarecomb4.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_recomb\\Datarecomb5.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_recomb\\Datarecomb6.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_recomb\\Datarecomb7.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_recomb\\Datarecomb8.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_recomb\\Datarecomb9.csv",
        ],
    ),
    ignore_index=True,
)

datas8_v2 = pd.concat(
    map(
        pd.read_csv,
        [
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_recombmuta\\Datarecomb_mut0.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_recombmuta\\Datarecomb_mut1.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_recombmuta\\Datarecomb_mut2.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_recombmuta\\Datarecomb_mut3.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_recombmuta\\Datarecomb_mut4.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_recombmuta\\Datarecomb_mut5.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_recombmuta\\Datarecomb_mut6.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_recombmuta\\Datarecomb_mut7.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_recombmuta\\Datarecomb_mut8.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_recombmuta\\Datarecomb_mut9.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_recomb\\Datarecomb0.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_recomb\\Datarecomb1.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_recomb\\Datarecomb2.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_recomb\\Datarecomb3.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_recomb\\Datarecomb4.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_recomb\\Datarecomb5.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_recomb\\Datarecomb6.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_recomb\\Datarecomb7.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_recomb\\Datarecomb8.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_recomb\\Datarecomb9.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_0.01\\Data0.010.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_0.01\\Data0.011.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_0.01\\Data0.012.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_0.01\\Data0.013.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_0.01\\Data0.014.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_0.01\\Data0.015.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_0.01\\Data0.016.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_0.01\\Data0.017.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_0.01\\Data0.018.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_0.01\\Data0.019.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_NN64_64\\DataNN_64_640.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_NN64_64\\DataNN_64_641.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_NN64_64\\DataNN_64_642.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_NN64_64\\DataNN_64_643.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_NN64_64\\DataNN_64_644.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_NN64_64\\DataNN_64_645.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_NN64_64\\DataNN_64_646.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_NN64_64\\DataNN_64_647.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_NN64_64\\DataNN_64_648.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_NN64_64\\DataNN_64_649.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_random\\Datarandom.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_random\\Datarandom1.csv",
        ],
    ),
    ignore_index=True,
)

# Seeds 0 - 9 with NN (2,1) as hidden units Mutationspower = 0.02 and 200 Generations
datas9_v2 = pd.concat(
    map(
        pd.read_csv,
        [
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_NN21\\DataNN_2_10.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_NN21\\DataNN_2_11.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_NN21\\DataNN_2_12.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_NN21\\DataNN_2_13.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_NN21\\DataNN_2_14.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_NN21\\DataNN_2_15.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_NN21\\DataNN_2_16.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_NN21\\DataNN_2_17.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_NN21\\DataNN_2_18.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_NN21\\DataNN_2_19.csv",
        ],
    ),
    ignore_index=True,
)

# Seeds 0 - 9 with NN (64,64) as hidden units Mutationspower = 0.02 and 200 Generations
datas10_v2 = pd.concat(
    map(
        pd.read_csv,
        [
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_NN64_64\\DataNN_64_640.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_NN64_64\\DataNN_64_641.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_NN64_64\\DataNN_64_642.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_NN64_64\\DataNN_64_643.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_NN64_64\\DataNN_64_644.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_NN64_64\\DataNN_64_645.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_NN64_64\\DataNN_64_646.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_NN64_64\\DataNN_64_647.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_NN64_64\\DataNN_64_648.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_NN64_64\\DataNN_64_649.csv",
        ],
    ),
    ignore_index=True,
)


# Randomebaseline
datas_random = pd.concat(
    map(
        pd.read_csv,
        [
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_random\\Datarandom.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_random\\Datarandom1.csv",
        ],
    ),
    ignore_index=True,
)

# Compare random with Mutationpower =  0.02 over 500 Agents and 200 Generations, 4 Layers and NN 64x64
data_BA1 = pd.concat(
    map(
        pd.read_csv,
        [
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_random\\Datarandom.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_random\\Datarandom1.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2\\Data0.020.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2\\Data0.021.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2\\Data0.022.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2\\Data0.023.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2\\Data0.024.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2\\Data0.025.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2\\Data0.026.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2\\Data0.027.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2\\Data0.028.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2\\Data0.029.csv",
        ],
    ),
    ignore_index=True,
)

# Compare different Mutationpowers
data_BA2 = pd.concat(
    map(
        pd.read_csv,
        [
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_0.01\\Data0.010.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_0.01\\Data0.011.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_0.01\\Data0.012.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_0.01\\Data0.013.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_0.01\\Data0.014.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_0.01\\Data0.015.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_0.01\\Data0.016.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_0.01\\Data0.017.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_0.01\\Data0.018.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_0.01\\Data0.019.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2\\Data0.020.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2\\Data0.021.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2\\Data0.022.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2\\Data0.023.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2\\Data0.024.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2\\Data0.025.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2\\Data0.026.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2\\Data0.027.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2\\Data0.028.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2\\Data0.029.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_0.03\\Data0.030.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_0.03\\Data0.031.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_0.03\\Data0.032.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_0.03\\Data0.033.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_0.03\\Data0.034.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_0.03\\Data0.035.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_0.03\\Data0.036.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_0.03\\Data0.037.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_0.03\\Data0.038.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_0.03\\Data0.039.csv",
        ],
    ),
    ignore_index=True,
)

# Compare differen reproduction options
data_BA3 = pd.concat(
    map(
        pd.read_csv,
        [
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2\\Data0.020.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2\\Data0.021.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2\\Data0.022.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2\\Data0.023.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2\\Data0.024.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2\\Data0.025.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2\\Data0.026.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2\\Data0.027.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2\\Data0.028.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2\\Data0.029.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_recombmuta\\Datarecomb_mut0.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_recombmuta\\Datarecomb_mut1.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_recombmuta\\Datarecomb_mut2.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_recombmuta\\Datarecomb_mut3.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_recombmuta\\Datarecomb_mut4.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_recombmuta\\Datarecomb_mut5.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_recombmuta\\Datarecomb_mut6.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_recombmuta\\Datarecomb_mut7.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_recombmuta\\Datarecomb_mut8.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_recombmuta\\Datarecomb_mut9.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_recomb\\Datarecomb0.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_recomb\\Datarecomb1.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_recomb\\Datarecomb2.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_recomb\\Datarecomb3.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_recomb\\Datarecomb4.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_recomb\\Datarecomb5.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_recomb\\Datarecomb6.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_recomb\\Datarecomb7.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_recomb\\Datarecomb8.csv",
            "C:\\Users\\felix\\Desktop\\BA-Daten_UNI\\Test_4Layer_v2_recomb\\Datarecomb9.csv",
        ],
    ),
    ignore_index=True,
)

sns.lineplot(
    data=datas,
    x="current_generation",
    y="coin_avg",
    hue="test",
    palette="colorblind",
)
# datas["avg_score"] = datas["avg_score"].ewm(span=5).mean()
sns.lineplot(
    data=datas,
    x="current_generation",
    y="own_coin_avg",
    hue="test",
    palette="colorblind",
)


print(datas)
plt.show()
