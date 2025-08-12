from ray import tune

search_space = {
    "seed": 42,
    "epochs": 100,
    "dataset_name": "RotMnist", 
    "model_name" : "PrototypeOptimalInvCNN",
    "m_param": {
        "in_channels" : 1,
        "input_size" : 56,
        "num_classes" : 10,
        "init_channels": tune.choice([4, 6, 8]), 
        "n_blocks": tune.choice([3, 4]),
        "m_layers": tune.choice([2, 3, 4]),
        "max_order": tune.choice([1, 2, 3, 4]),
        "circular_padding": "tukey"

    },
    "optimizer_hparams": {
            "lr": 0.01,
    }, 
    "optimizer_name": "AdamW",
    "dataset_hparams": {
        "batch_size": 16,
        "data_dir" : "/vast/home/karella/rotation-invariant-neural-networks/hippy2d/data",
        "to_complex" : False,
        "augment" : True 
    },
    "lr_name": "MultiStepLR",
    "lr_hparams": {
        "milestones": [30, 80],
        "gamma": 0.1
    }
}
