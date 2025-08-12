from ray import tune

search_space = {
    "seed": 42,
    "epochs": 100,
    "dataset_name": "RESISC45", 
    "model_name" : "PrototypeOptimalInvCNN",
    "m_param": {
        "in_channels" : 3,
        "input_size" : 256,
        "num_classes" : 45,
        "init_channels": tune.choice([4, 8]), 
        "n_blocks": tune.choice([3, 4]),
        "m_layers": tune.choice([2, 3, 4]),

    },
    "optimizer_hparams": {
            "lr": 0.01,
    }, 
    "optimizer_name": "AdamW",
    "dataset_hparams": {
        "batch_size": 8,
        "data_dir" : "/vast/home/karella/rotation-invariant-neural-networks/hippy2d/data",
        "to_complex" : False, 
    },
    "lr_name": "MultiStepLR",
    "lr_hparams": {
        "milestones": [30, 80],
        "gamma": 0.1
    }
}
