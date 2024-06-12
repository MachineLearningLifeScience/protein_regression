from gpflow.kernels import Matern52, SquaredExponential

from algorithms import KNN, GPonRealSpace, RandomForest, UncertainRandomForest
from util.mlflow.constants import (AT_RANDOM, ESM, ESM1V, ESM2, EVE,
                                   EVE_DENSITY, ONE_HOT, PROTT5, PSSM, ROSETTA,
                                   TRANSFORMER, VAE_AUX, VAE_DENSITY, VAE_RAND)
from visualization.plot_metric_for_mixtures import (
    plot_metric_against_threshold, plot_metric_for_mixtures)

colorscheme = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52", "darkred", "darkcyan", "darkblue"]

colorscheme_alternate = ['purple', 'chocolate', 'lightgreen', 'indigo', 'orange', 'darkblue', 'cyan', 'olive', 'brown', 'pink', 'darkred', 'dimgray', 'blue', 'darkorange', 'k', 'lightblue', 'green']

colorscheme_reds = ["#781414", "#C01717", "#BD3434", "#EC7474", "#FF5000", "#800000"]
colorscheme_blues = ["#78ACC8", "#233757", "#4F6A93", "#2A4B7B", "#35465E", "#22EAEA"]


representation_colors = {ONE_HOT: colorscheme[0],
            EVE: colorscheme[1],
            EVE_DENSITY: colorscheme[2],
            TRANSFORMER: colorscheme[3],
            ESM: colorscheme[4],
            EVE_DENSITY: colorscheme[5],
            "additive": colorscheme[6],
            PSSM: colorscheme[7],
            ESM1V: colorscheme[8],
            ESM2: colorscheme[9],
            PROTT5: colorscheme[10],
        }


unsupervised_reference_colors = {
    EVE: colorscheme[1],
    "deepsequence": colorscheme[11],
    "tranception": colorscheme[12],
}

representation_markers = {ONE_HOT: "o",
                        EVE: "D",
                        TRANSFORMER: "^",
                        ESM: "v",
                        EVE_DENSITY: "8"}

augmentation_colors = {VAE_DENSITY: 'pink', 
                    EVE_DENSITY: 'cornflowerblue', 
                    ROSETTA: 'lightgreen'
                }

algorithm_colors = {a: colorscheme[i] for i, a in enumerate([
    GPonRealSpace().get_name(),
    'GPsqexp',
    GPonRealSpace(kernel_factory= lambda: Matern52()).get_name(),
    RandomForest().get_name(),
    KNN().get_name(),
    UncertainRandomForest().get_name(),
    EVE_DENSITY,
    AT_RANDOM,
])}

algorithm_markers = {
    GPonRealSpace().get_name(): "<",
    'GPsqexp': "^",
    GPonRealSpace(kernel_factory= lambda: Matern52()).get_name(): ">",
    UncertainRandomForest().get_name(): "P",
    RandomForest().get_name(): "X", 
    KNN().get_name(): "o", 
    EVE_DENSITY: "8",
}

task_colors = {"RandomSplitter": colorscheme[0],
            "PositionSplitter_p15": colorscheme[1],
            "Fractional": colorscheme[2],
            "Optimization": colorscheme[3]}


task_colors_to_algos_ablation = {"Random": {a: colorscheme_blues[i] for i, a in enumerate([
                GPonRealSpace().get_name(),
                'GPsqexp',
                GPonRealSpace(kernel_factory= lambda: Matern52()).get_name(),
                RandomForest().get_name(),
                KNN().get_name(),
                UncertainRandomForest().get_name()
                ])
            }, 
            "Position": {a: colorscheme_reds[i] for i, a in enumerate([
                GPonRealSpace().get_name(),
                'GPsqexp',
                GPonRealSpace(kernel_factory= lambda: Matern52()).get_name(),
                RandomForest().get_name(),
                KNN().get_name(),
                UncertainRandomForest().get_name()
                ])
            }
    }