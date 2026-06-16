import itertools
import torch
import timm
import torchvision.transforms as T

from torch.optim import SGD
from wildlife_datasets.datasets import BrownBearHeads
from wildlife_tools.train import ArcFaceLoss, BasicTrainer, set_seed

def main():
    root = "data/BrownBearHeads"
    BrownBearHeads.get_data(root)

    set_seed(666)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = T.Compose([
        T.Resize((384, 384)),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    dataset = BrownBearHeads(
    root,
    transform=transform,
    load_label=True,
    factorize_label=True,
    )
    
    # Use prepared bear split
    split = "split_iid"
    idx_train = dataset.df.index[dataset.df[split] == "train"].tolist()
    dataset_train = dataset.get_subset(idx_train)
    
    backbone = timm.create_model(
    "hf-hub:BVRA/MegaDescriptor-L-384",
    num_classes=0,
    pretrained=True,
    )
    
    with torch.no_grad():
        x, _ = dataset_train[0]
        embedding_size = backbone(x.unsqueeze(0)).shape[1]
    
    objective = ArcFaceLoss(
    num_classes=dataset_train.num_classes,
    embedding_size=embedding_size,
    margin=0.5,
    scale=64,
    )
    
    params = itertools.chain(backbone.parameters(), objective.parameters())
    
    optimizer = SGD(
    params=params,
    lr=0.001,
    momentum=0.9,
    )
    
    trainer = BasicTrainer(
    dataset=dataset_train,
    model=backbone,
    objective=objective,
    optimizer=optimizer,
    epochs=10,
    batch_size=8,
    device=device,
    )
    
    trainer.train()
    
    torch.save(
    {
    "model": backbone.state_dict(),
    "objective": objective.state_dict(),
    "labels_map": dataset_train.labels_map,
    },
    "brown_bear_megadescriptor_arcface.pt",
    )


if __name__ == "__main__":
    main()
