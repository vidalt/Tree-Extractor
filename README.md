# Tree-Extractor

This repository contains the code for the paper "From Counterfactuals to Trees: Competitive Analysis of Model Extraction Attacks". Arxiv preprint available [here](https://arxiv.org/abs/2502.05325).

## OCEAN and Gurobi solver
OCEAN is a library for generating counterfactual explanations. The library is available [here](https://github.com/vidalt/OCEAN). 
In our code, we use a slightly modified version of OCEAN to accelerate the generation of counterfactuals. The modified version is available [here](https://github.com/AwaKhouna/OCEAN.git). OCEAN requires the Gurobi solver to be installed. The Gurobi solver can be downloaded from https://www.gurobi.com/downloads/.

## Getting Started
### Requirements
Python 3.10.0 or later with all requirements.txt dependencies installed. To install run:
```bash
$ # Create a virtual environment
$ python -m venv myenv
$ source myenv/bin/activate
$ pip install -r requirements.txt
$ # Install OCEAN library
$ git clone https://github.com/AwaKhouna/OCEAN.git
$ cd OCEAN
$ pip install -e .
$ cd ..
```

### Simple Example
To run a simple example of the `TRA` attack on the `COMPAS` dataset, run:
```python
import numpy as np
from attacks.TreeReconstructionAttack import TRAttack
from sklearn.ensemble import RandomForestClassifier
from OCEAN.src.DatasetReader import DatasetReader
from utils.CounterFactualExp import CounterFactualOracle

datasetPath = "datasets/COMPAS-ProPublica_processedMACE.csv"
reader = DatasetReader(datasetPath, SEED=42)
# put the acctionability to FREE for all the features
reader.featuresActionnability = np.array(["FREE"] * len(reader.featuresActionnability))

# Train a random forest using sklearn
rf = RandomForestClassifier(n_estimators=3, random_state=42)
rf.fit(reader.X_train.values, reader.y_train.values)

# Initialize the oracle and the attacker
Oracle = CounterFactualOracle(
    rf,
    reader.featuresType,
    reader.featuresPossibleValues,
    norm=2,
    n_classes=2,
    SEED=42,
)
attacker = TRAttack(
    Oracle,
    FeaturesType=reader.featuresType,
    FeaturesPossibleValues=reader.featuresPossibleValues,
    ObjNorm=2,
    strategy="BFS",
)

# Run the attack
attacker.attack()

# Compute the fidelity of the extracted model
fidelity = attacker.compute_fidelity(Oracle.classifier, reader.X_test.values)

print(f"Fidelity: {fidelity*100:.2f}%")
```

expected output:
```bash
Nb queries:  349
Fidelity: 100.00%
```

## Code structure
- `datasets/` contains the datasets used in the experiments.
- `experiments/` :
  - `plots/` contains the plots generated in the experiments.
  - `results/` contains the results of the decision trees experiments in json format.
  - `RFs/` contains the random forests results in the experiments in json format.
  - `exp_res.py` contains the code to generate the trees plots.
  - `expRF_res.py` contains the code to generate the random forests plots.
  - `parameters.py` contains the parameters used in the experiments.
- `attacks/` :
  - `StealML/` contains the code for the `PathFinding` (Tramèr et al., 2016) attack taken from : https://github.com/ftramer/Steal-ML.
  - `SurrogateAttacks.py` contains the code for the surrogate attacks `CF` (Aïvodji et al., 2020) and `DualCF` (Wang et al., 2022) algorithms.
  - `TreeReconstructionAttack.py` contains the code for our proposed the tree reconstruction (`TRA`) attack algorithm.
- `OCEAN/` contains a fork version of the OCEAN library.
- `utils/` :
  -  `CounterFactualExp.py` contains an oracle class to generate counterfactuals using the OCEAN library.
  -  `DiCEOracle.py` contains an oracle class to generate counterfactuals using the [DiCE](https://github.com/interpretml/DiCE) library.
  -  `ExtractedTree.py` contains the code to generate the decision trees from the extracted models.
  -  `NodeIdOracle.py` contains an oracle class to generate the node ids of the decision trees to use for the `PathFinding` attack.
- `experiment.py` contains the code to run the decision trees experiments.
- `experimentRF.py` contains the code to run the random forests experiments.



## Reproducing the paper experiments
### Decision Trees experiments
The experiments are labeled using the `--experiment` flag. 
To run the decision trees experiments, run:
```bash
$ python experiment.py --experiment=$EXPERIMENTNUMBER
```
where `$EXPERIMENTNUMBER` is the number of the experiment to run. The experiments are labeled as follows: `$EXPERIMENTNUMBER = DatasetIndex * 20 + AttackIndex * 5 + SeedIndex + 1`. For example, to run the experiment with the `seed=2` on the `COMPAS` dataset using the `TRA` attack, set `$EXPERIMENTNUMBER = 1 * 20 + 0 * 5 + 0 + 1= 21`.

```bash
$ python experiment.py --experiment=21
```
In total there are 5 (Datasets) * 4 (Attackers) * 5 (Seeds) = 100 experiments. The results of the experiments are saved directly in the `experiments/results/` folder.

### Random Forests experiments
To run the random forests experiments, run:
```bash
$ python experimentRF.py --experiment=$EXPERIMENTNUMBER
```
where `$EXPERIMENTNUMBER` is the number of the experiment to run. The experiments are labeled same as for decision trees experiments but there is only one dataset. For example, to run the experiment with the `seed=31` on the `COMPAS` dataset using the `TRA` attack, set `$EXPERIMENTNUMBER = 0 * 3 + 1 + 1= 2`.

```bash
$ python experimentRF.py --experiment=2
```

### Plotting the results
To plot the results of the decision trees experiments, run:
```bash
$ python experiments/exp_res.py
```
To plot the results of the random forests experiments, run:
```bash
$ python experiments/expRF_res.py
```


## References

- Tramèr, F., Zhang, F., Juels, A., Reiter, M. K., & Ristenpart, T. (2016). Stealing machine learning models via prediction {APIs}. In 25th USENIX security symposium (USENIX Security 16) (pp. 601-618).
- Aïvodji, U., Bolot, A., & Gambs, S. (2020). Model extraction from counterfactual explanations. arXiv preprint arXiv:2009.01884.
- Yongjie Wang, Hangwei Qian, and Chunyan Miao. 2022. DualCF: Efficient Model Extraction Attack from Counterfactual Explanations. In Proceedings of the 2022 ACM Conference on Fairness, Accountability, and Transparency (FAccT '22). Association for Computing Machinery, New York, NY, USA, 1318–1329. https://doi.org/10.1145/3531146.3533188
