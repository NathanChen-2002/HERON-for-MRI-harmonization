def get_dataset_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


class ssl_dataset():
    def __init__(self):
        super(ssl_dataset, self).__init__()
        # data parameters
        self.channel_dim = 0
        self.cache_rate = 0.2
        self.max_graylevel = 1.5134999752044678 # From preprocess if trained
        self.min_graylevel = 0.0 # From preprocess if trained
        self.size_x = 192
        self.size_y = 192
        self.cutlowbond = 3  
        self.cuthighbond = 156

        # model configs
        self.input_channels = 3
        self.site_input_channels = 1
        self.site_output_channels = 1
        self.decoder_output_channels = 1




