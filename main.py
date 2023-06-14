from datasets import load_dataset
from torch.utils.data import DataLoader

dataset = load_dataset("axiong/pmc-oa")
train_dataset = dataset['train']
train_dataloader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    # num_workers=4,
    # prefetch_factor=2,
    # persistent_workers=True,
    # pin_memory=True,
    drop_last=False,
)
for batch in train_dataloader:
    print(batch)
    break