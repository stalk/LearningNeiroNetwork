import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import model
import train
import test
from helloWorld.first import device, SimpleNN

batch_size = 64
training_data = datasets.MNIST(
    root="data",
    train=True,
    transform=ToTensor(),
    download=True
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    transform=ToTensor(),
    download=True
)

train_data_loader = DataLoader(training_data,batch_size=batch_size)
test_dat_loader = DataLoader(test_data,batch_size=batch_size)

device ='cuda' if torch.cuda.is_available() else 'cpu'
print(f"using device: {device}")

model = SimpleNN().to(device)
loss_fn = nn.CrossEntropyLoss()
learning_rate = 1e-2
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

epochs = 20

for epoch in range(epochs):
    print(f'Epoch {epoch+1} \n ----------------------------------')
    train.train_loop(train_data_loader, model, loss_fn, optimizer, device)
    test.test_loop(test_dat_loader, model, loss_fn,device)


print("Done!")
