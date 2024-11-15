import torch 

from .Merges import Merges

from peft import get_peft_model, set_peft_model_state_dict

class FlanT5Avg(Merges):
    def __init__(self, name):
        super().__init__(name)


        '''
        These values are meant to be modified by the user.
        '''
        # Give a list of models to load for the merge 
        self.list_models = [("jash6/ft-t5-small", "613ed52638fb5e53cb6feb9cdc1619749a700fd9"), 
                            ("jash6/ft-t5-small-2", "ac93ea89ad7dfb4612790140798d50e7e83c5dfe")]
        
        # Hyperparameters 
        self.base_model_name = "google/flan-t5-small"
        self.base_model_revision_id = "0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab"
        self.is_peft = True
        self.max_seq_len = 512
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Architecture must match base model. 
        self.architecture = "encoder_decoder"

        '''
        These are variables used later in the code and not intended to be set, but feel free to adapt to your use case.  
        '''
        # Loaded models and configs 
        self.loaded_models = {}
        self.loaded_configs = {}

        # Merged model parameters
        self.merged_model = {}

    # Implement merge function 
    def merge(
        self,
    ):

        '''
        1) Load HuggingFace checkpoints and configs 
        '''
        super()._load_huggingface_models_and_configs()

        '''
        2) Merge checkpoints  
        '''
        parameter_lambdas = [0.5, 0.5]

        # Get individual models 
        all_models = list(self.loaded_models.values())

        # Get all the parameters names (uses the first model and assume all the models have the same parameter)
        all_parameter_names = all_models[0].keys()

        for parameter_name in all_parameter_names:
            merged_parameter = None
            for parameter_lambda, model in zip(parameter_lambdas, all_models):
                parameter = model[parameter_name]
                if merged_parameter is None:
                    merged_parameter = torch.clone(parameter) * parameter_lambda
                else:
                    merged_parameter += parameter * parameter_lambda
            self.merged_model[parameter_name] = merged_parameter
        '''
        3) Load base model and tokenizer 
        '''
        self._load_base_model()
        self._load_tokenizer()

        '''
        4) Load merged model into base model 
        '''
        # Modify the base model. This is needed for Peft, which wraps the base_model in a Peft wrapper. 
        huggingface_config = list(self.loaded_configs.values())[0]
        if huggingface_config is not None:
            self.base_model = get_peft_model(self.base_model, huggingface_config)
            set_peft_model_state_dict(self.base_model, self.merged_model)
        
        else:
            self.base_model.load(self.merged_model)

        # Requires to make results deterministic. If not set, we will just run once and use the results from the first pass. 
        self.base_model.eval()

        return self.base_model