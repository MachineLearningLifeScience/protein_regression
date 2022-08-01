from util.mlflow.constants import ONE_HOT, ESM, TRANSFORMER, EVE, VAE_AUX, VAE_RAND
from util.mlflow.constants import ROSETTA, VAE_DENSITY, EVE_DENSITY
from gpflow.kernels import SquaredExponential, Matern52
from algorithms import GPonRealSpace, RandomForest, KNN, UncertainRandomForest

colorscheme = ['dimgrey', '#661100', '#332288', 'hotpink',  "cyan", '#117733', "lime", "tan", "orangered"]

colorscheme_alternate = ['purple', 'chocolate', 'lightgreen', 'indigo', 'orange', 'darkblue', 'cyan', 'olive', 'brown', 'pink', 'darkred', 'dimgray', 'blue', 'darkorange', 'k', 'lightblue', 'green']

representation_colors = {ONE_HOT: colorscheme[0],
            EVE: colorscheme[1],
            VAE_AUX: colorscheme[2],
            TRANSFORMER: colorscheme[3],
            ESM: colorscheme[4]}

augmentation_colors = {VAE_DENSITY: 'pink', EVE_DENSITY: 'cornflowerblue', ROSETTA: 'lightgreen'}

algorithm_colors = {a: colorscheme[i] for i, a in enumerate([GPonRealSpace().get_name(), 
                'GPsqexp', 
                GPonRealSpace(kernel_factory= lambda: Matern52()).get_name(),
                UncertainRandomForest().get_name(),
                RandomForest().get_name(), 
                KNN().get_name(), 
                EVE_DENSITY])}