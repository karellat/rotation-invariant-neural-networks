# Inference with a pretrained model
import numpy as np
import timm
import torch
import torchvision.transforms as T

from wildlife_datasets.datasets import BrownBearHeads
from wildlife_tools.features import DeepFeatures
from wildlife_tools.similarity import CosineSimilarity
from wildlife_tools.inference import KnnClassifier

def main():

    root = "data/BrownBearHeads"
    BrownBearHeads.get_data(root)

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

    split = "split_iid" # or "split_ood"
    idx_train = dataset.df.index[dataset.df[split] == "train"].tolist()
    idx_test = dataset.df.index[dataset.df[split] == "test"].tolist()


    database = dataset.get_subset(idx_train)
    query = dataset.get_subset(idx_test)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    backbone = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384",
                                num_classes=0,
                                pretrained=True)

    extractor = DeepFeatures(backbone,
                            batch_size=96,
                            device=device)
    query_features = extractor(query)
    database_features = extractor(database)

    similarity = CosineSimilarity()(query_features, database_features)
    print(similarity.keys())
    similarity = similarity["cosine"]

    classifier = KnnClassifier(k=1, database_labels=database.labels_string)
    predictions = classifier(similarity)

    accuracy = np.mean(query.labels_string == predictions)
    print(accuracy)

if __name__ == "__main__":
    main()