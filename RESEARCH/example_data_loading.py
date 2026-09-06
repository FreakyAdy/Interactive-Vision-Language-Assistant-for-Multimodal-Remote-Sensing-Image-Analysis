from ben_txt_datamodule import BENTxTDataset, BENTxTDataModule

def create_dataset_example():
    # Datasets example using the Red (B04), Green (B03), and Blue (B02) band from the Sentinel-2 images.
    ds_rgb = BENTxTDataset(
        lmdb_file = "Encoded-BigEarthNet/",
        metadata_file = "BigEarthNet.txt.parquet",
        bands = ("B04", "B03", "B02"),
        img_size = 120
    )

    sample = ds_rgb[0]
    print(f"RGB input image: {sample['image_input'].shape}")
    print(f"Text input: {sample['text_input']}")
    print(f"Reference output: {sample['reference_output']}")

def create_datamodule_example():
    # Lightning DataModule example using the 10m and 20m spatial resolution bands from Sentinel-1 and Sentinel-2 and multiple metadata filters.
    # The datamodule will create 4 dataloaders: train, val, test, and bench.
    dm = BENTxTDataModule(
        image_lmdb_file = "Encoded-BigEarthNet/",
        metadata_file = "BigEarthNet.txt.parquet",
        bands = 'S1S2-10m20m',
        img_size = 120,
        batch_size = 1,
        num_workers_dataloader = 0,
        types = ['mcq'],
        categories = ['climate zone'],
        countries = ['Portugal', 'Finland'],
        seasons = ['Summer'],
        climate_zones = None,
        point_token = ['<point>', '</point>'],
        ref_token = ['<ref>', '</ref>']
    )
    dm.setup()
    
    train_dl = dm.train_dataloader()
    for batch in train_dl:
        print(f"Batch image input shape: {batch['image_input'].shape}")
        print(f"First batch sample text input: {batch['text_input'][0]}")
        print(f"First batch sample text reference output: {batch['reference_output']}")
        break

if __name__ == "__main__":
    create_dataset_example()
    create_datamodule_example()