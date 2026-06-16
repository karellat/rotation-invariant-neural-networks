
def main():
    print("Hello from wildlife!")
    from wildlife_datasets import datasets

    # Pose estimation with a pretrained model
    root = "data/BrownBearHeads"

    datasets.BrownBearHeads.get_data(root)

    dataset = datasets.BrownBearHeads(root, img_load="full")
    print(dataset.df.head())

if __name__ == "__main__":
    main()