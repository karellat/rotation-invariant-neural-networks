from ray import tune

search_space = {
    "seed": 42,
    "epochs": 100,
    "dataset_name": "MnistRotTest", 
    "model_name" : "PrototypeOptimalInvCNN",
    "m_param": {
        "in_channels" : 1,
        "input_size" : 64,
        "num_classes" : 10,
        "max_order": tune.choice([1, 2, 3, 4]),
        "filter_size" : tune.choice([5, 7, 9, 11, 13, 15]),
        "circular_padding": "tukey"
    },
    "optimizer_hparams": {
            "lr": 0.01525,
    }, 
    "optimizer_name": "AdamW",
    "dataset_hparams": {
        "batch_size": 128,
        "data_dir" : "/vast/home/karella/rotation-invariant-neural-networks/hippy2d/data",
        "to_complex" : False,
    },
    "lr_name": "MultiStepLR",
    "lr_hparams": {
        "milestones": [30, 80],
        "gamma": 0.1
    }
}
