class ImputationManagement:
    """
    Factory class for managing custom imputation methods.
    
    This class allows users to register and use custom imputation functions.
    For built-in methods, use the 'gainpro' command-line tool instead:
    - gainpro median: for median imputation
    - gainpro download: for HuggingFace models
    """
    def __init__ (self, model, df_missing, missing_file_path):
        self.model = model
        self.df = df_missing
        self.missing = missing_file_path
        self.dict_imputation_methods = {}


    def add_method(self, model, fn): 
        if model in self.dict_imputation_methods:
            raise SystemExit ("Method already exists")
        else:
            self.dict_imputation_methods.update({model:fn})


    def run_model(self, model):
        if model not in self.dict_imputation_methods:
            raise SystemExit (f"Unknown model called {model}, Models available are: {','.join(self.dict_imputation_methods)}")
        else:
            method = self.dict_imputation_methods[model]
            # Handle both callable functions and objects with .run() method
            if callable(method) and not hasattr(method, 'run'):
                # It's a plain function, call it directly
                return method(self.df)
            elif hasattr(method, 'run'):
                # It's an object with a .run() method
                return method.run(self.df)
            else:
                raise SystemExit(f"Invalid imputation method: {model} must be callable or have a .run() method")










