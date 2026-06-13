import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print("device:", device)
print("torch:", torch.__version__)

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
ds = datasets.ImageFolder("data/Dataset", transform=tf)
print("classes:", ds.classes)
print("class_to_idx:", ds.class_to_idx)
print("total images:", len(ds))

loader = DataLoader(ds, batch_size=8, shuffle=True)
x, y = next(iter(loader))
print("batch x shape:", x.shape)
print("batch y:", y)