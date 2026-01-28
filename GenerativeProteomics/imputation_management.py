from models.base_abstract import ImputationModel
from models.gain_dann import GainDannImputationModel
from models.medium import MediumImputationModel
from models.missforest import MissForestRImputationModel
from models.mice import IterativeMICEImputationModel


class Imputation_Management:
    def __init__ (self, model, df_missing, missing_file_path):
        self.model = model
        self.df = df_missing
        self.missing = missing_file_path
        self.dict_imputation_methods = {"GAIN_DANN_model" : GainDannImputationModel(), "medium_imputation" : MediumImputationModel(), "missforest": MissForestRImputationModel(), "mice": IterativeMICEImputationModel()}


    def add_method(self, model, fn): 
        if model in self.dict_imputation_methods:
            raise SystemExit ("Method already exists")
        else:
            self.dict_imputation_methods.update({model:fn})


    def run_model(self, model):
        if model not in self.dict_imputation_methods:
            raise SystemExit (f"Unknown model called {model}, Models available are: {','.join(self.dict_imputation_methods)}")
        else:
            return self.dict_imputation_methods[model].run(self.df)










