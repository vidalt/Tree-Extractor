# These are colorblindness-friendly colors taken from this link:
# https://medium.com/@allieofisher/inclusive-color-palettes-for-the-web-bbfe8cf2410e
color_cycle = [
    "#BDD9BF",
    "#929084",
    "#FFC857",
    "#A997DF",
    "#E5323B",
    "#2E4052",
    "#6FDE6E",
]


# Datasets disponibles
DATASETS = [
    "datasets/Adult_processedMACE.csv",
    "datasets/COMPAS-ProPublica_processedMACE.csv",
    "datasets/Credit-Card-Default_processedMACE.csv",
    "datasets/German-Credit.csv",
    "datasets/Students-Performance-MAT.csv",
]

SHORTNAMES = [
    "Adult",
    "COMPAS",
    "CreditCard",
    "GCredit",
    "SPerformanceMAT",
]

ATTACKS = [
    "TRA",
    "DualCF",
    "CF",
    "PathFinding",
]

# The seeds has been chosen to be prime numbers :-)
SEEDS = [
    2,
    31,
    73,
    127,
    179,
]

DEPTH = [i for i in range(4, 11)] + [None]

MARKERS = {
    "PathFinding": "x",
    "DualCF_DT": "o",
    "DualCF_DT+": "o",
    "DualCF_MLP": "o",
    "TRA": "p",
    "CF_DT": "v",
    "CF_DT+": "v",
    "CF_MLP": "v",
}

# Assign colors to categories with variations
COLORS = {
    "PathFinding": color_cycle[6],
    "DualCF_DT": color_cycle[1],
    "DualCF_DT+": color_cycle[2],
    "DualCF_MLP": color_cycle[3],
    "TRA": color_cycle[4],
    "CF_DT": color_cycle[5],
    "CF_DT+": color_cycle[2],
    "CF_MLP": color_cycle[3],
}

# Assign colors to categories with variations
COLORS_RF = {
    "PathFinding": color_cycle[0],
    "DualCF_RF": color_cycle[1],
    "DualCF_RF+": color_cycle[2],
    "DualCF_MLP": color_cycle[3],
    "TRA": color_cycle[4],
    "CF_RF": color_cycle[5],
    "CF_RF+": color_cycle[2],
    "CF_MLP": color_cycle[3],
}
MARKERS_RF = {
    "DualCF_RF": "o",
    "DualCF_RF+": "o",
    "DualCF_MLP": "o",
    "TRA": "p",
    "CF_RF": "v",
    "CF_RF+": "v",
    "CF_MLP": "v",
}


DATASET_RF = "datasets/COMPAS-ProPublica_processedMACE.csv"

ATTACKS_RF = [
    "TRA",
    "DualCF",
    "CF",
]

ESTIMATORS = [
    5,
    25,
    50,
    75,
    100,
]
