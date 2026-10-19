def get_hparams_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(dataset_name))
    return globals()[dataset_name]


class VICReg():
    def __init__(self):
        super(VICReg, self).__init__()
        self.train_params = {
            'num_epochs': 100,
            'batch_size':400,
            'weight_decay': 1e-4,    
            'learning_rate': 1e-4,
            "proj_hidden_dim": 128,
            "mc_k": 144,
        }
        self.alg_hparams_ssl_dataset = {
            'HERON': {'filter_size': [32,64,128,256],
                        'stride_encoder': 2, 
                         }
        }
        self.other_hparams= {                     
            'brain':{"dim":32,
                     "L_inv_weight":25,
                     "L_var_weight":12.5,
                     "L_cov_weight":1,
                     "L_site_mean_weight": 12.5, 
                     "L_site_mean_lambda": 0, 
                     "std_gamma": 0.3,  
                     "warmup_epochs":10,
                     "scheduler_start_factor": 1e-4,
                     "scheduler_min_factor": 1e-3,
                     "min_sites": 6}
        }


class MultiscalePretraining():
    def __init__(self):
        super(MultiscalePretraining, self).__init__()
        self.train_params = {
            'num_epochs': 60,
            'batch_size': 256,
            'weight_decay': 1e-4,
            'learning_rate': 2e-4,
        }
        self.alg_hparams_ssl_dataset = {
            'HERON': {
                'filter_size': [32,64,128,256],
                'stride_encoder': 2,
            }
        }
        self.other_hparams = {
            'multiscale': {
                "warmup_epochs": 3,
                "scheduler_start_factor": 0.5,
                "scheduler_min_factor": 0.05,
                "min_sites": 6,
            }
        }




class HarmonizationTraining():
    def __init__(self):
        super(HarmonizationTraining, self).__init__()
        self.train_params = {
            'num_epochs': 100,
            'batch_size': 45,
            'weight_decay': 1e-4,
            'learning_rate': 1e-4,
        }
        self.alg_hparams_ssl_dataset = {
            'HERON': {
                'filter_size': [32,64,128,256],
                'stride_encoder': 2,
            }
        }
        self.other_hparams = {
            'all': {
                "warmup_epochs": 10,
                "scheduler_start_factor": 0.1,
                "scheduler_min_factor": 0.05,
                "L_ir_weight": 10.0,
                "L_rs_weight": 30.0,
                "L_site_prototype_weight": 50.0,
                "L_id_weight": 20.0,
                "L_fr_weight": 70.0,
                "num_transformer_layer": 1,
                "min_sites": 2,
                "max_unique_sites_per_batch": 3,
                "min_samples_per_site_in_batch": 15,
            }
        }




