# search_space.py
from ray import tune

search_space = {
    "model_name" : "PrototypeOptimalInvCNN",
    "lr": tune.loguniform(1e-4, 1e-1),
    "dataset_name" : "MnistRotTest" 
}