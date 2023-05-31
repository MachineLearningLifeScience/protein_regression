from util.mlflow.constants import ONE_HOT, ESM, TRANSFORMER, EVE, VAE_AUX, VAE_RAND
from util.mlflow.constants import ROSETTA, VAE_DENSITY, EVE_DENSITY
from gpflow.kernels import SquaredExponential, Matern52
from algorithms import GPonRealSpace, RandomForest, KNN, UncertainRandomForest
from visualization.plot_metric_for_mixtures import plot_metric_for_mixtures
from visualization.plot_metric_for_mixtures import plot_metric_against_threshold

colorscheme = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"]

#colorscheme = ['dimgrey', '#661100', '#332288', '#054709',  "#126160", "crimson", "darkblue", "darkmagenta", "dimgray"]

colorscheme_alternate = ['purple', 'chocolate', 'lightgreen', 'indigo', 'orange', 'darkblue', 'cyan', 'olive', 'brown', 'pink', 'darkred', 'dimgray', 'blue', 'darkorange', 'k', 'lightblue', 'green']

representation_colors = {ONE_HOT: colorscheme[0],
            EVE: colorscheme[1],
            VAE_AUX: colorscheme[2],
            TRANSFORMER: colorscheme[3],
            ESM: colorscheme[4],
            EVE_DENSITY: colorscheme[5],
            "additive": colorscheme[6]}


task_colors = {"RandomSplitter": colorscheme[0],
            "PositionSplitter_p15": colorscheme[1],
            "Fractional": colorscheme[2],
            "Optimization": colorscheme[3]}


representation_markers = {ONE_HOT: "o",
                        EVE: "D",
                        TRANSFORMER: "^",
                        ESM: "v",
                        EVE_DENSITY: "8"}

augmentation_colors = {VAE_DENSITY: 'pink', EVE_DENSITY: 'cornflowerblue', ROSETTA: 'lightgreen'}

algorithm_colors = {a: colorscheme[i] for i, a in enumerate([GPonRealSpace().get_name(), 
                'GPsqexp', 
                GPonRealSpace(kernel_factory= lambda: Matern52()).get_name(),
                UncertainRandomForest().get_name(),
                RandomForest().get_name(), 
                KNN().get_name(), 
                EVE_DENSITY])}

algorithm_markers = {
    GPonRealSpace().get_name(): "<",
    'GPsqexp': "^",
    GPonRealSpace(kernel_factory= lambda: Matern52()).get_name(): ">",
    UncertainRandomForest().get_name(): "P",
    RandomForest().get_name(): "X", 
    KNN().get_name(): "o", 
    EVE_DENSITY: "8",
}