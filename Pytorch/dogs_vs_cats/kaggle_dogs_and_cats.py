import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

# Definições do dispositivo e dos caminhos das pastas (supondo que já estejam definidos)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dogs_train_folder_path = "/home/edvsimoes/Downloads/dogs-vs-cats/train/dogs"
cats_train_folder_path = "/home/edvsimoes/Downloads/dogs-vs-cats/train/cats"
database_train_folder = "/home/edvsimoes/Downloads/dogs-vs-cats/train"

# Transforms e DataLoader
transformations = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
])
'''
def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')'''

training_data_sets = datasets.ImageFolder(
    database_train_folder,
    transform=transformations,
)

training_data_loader = DataLoader(training_data_sets, batch_size=32, shuffle=True, num_workers=4)

# Definição do modelo
class Dogs_vs_cats_predicter_untrained(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.convo1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.pooling1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.convo2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.pooling2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.convo3 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1)

        # Calcula o tamanho de entrada para a camada Linear
        self.conv_output_size = self._get_conv_output_size((128, 128))
        

        self.linear1 = torch.nn.Linear(self.conv_output_size, 512, bias=True)
        self.linear2 = torch.nn.Linear(512, 2, bias=True)

        self.dropout = torch.nn.Dropout(p=0.1)

    def _get_conv_output_size(self, input_size):
        # Use a rede convolucional para calcular o tamanho da saída
        x = torch.zeros(1, 3, *input_size)
        x = self.pooling1(self.relu(self.convo1(x)))
        x = self.pooling2(self.relu(self.convo2(x)))
        x = self.relu(self.convo3(x))
        return np.prod(x.size()[1:])

    def forward(self, x):
        x = self.dropout(self.pooling1(self.relu(self.convo1(x))))
        x = self.dropout(self.pooling2(self.relu(self.convo2(x))))
        x = self.relu(self.convo3(x))
        x = x.view(-1, self.conv_output_size)
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x

# Teste do modelo
data = iter(training_data_loader)
xs, ys = next(data)
print(xs.size())

dogs_vs_cats_predicter = Dogs_vs_cats_predicter_untrained()
print(dogs_vs_cats_predicter(xs[0]))

def init_weights(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)

dogs_vs_cats_predicter.apply(init_weights)


def trainModel(f, dl, num_ephocs=1):
    dogs_vs_cats_predicter.to(device)
    optim = torch.optim.SGD(f.parameters(), lr = 0.007)
    error = torch.nn.CrossEntropyLoss()



    losses= []
    ephocs= []
    N = len(dl)
    print(N)
    for e in range(num_ephocs):
        data = iter(dl)
        print(f"Ephoc: {e}",flush=True)
        for i in range(500):
            x, y = next(data)
            x, y = x.to(device), y.to(device)
            print(i, end=" ",flush=True)
            optim.zero_grad()
            loss = error(f(x),y)
            loss.backward()
            optim.step()#litterally backporpagating

            ephocs.append(e+i/N)
            losses.append(loss.item())
    return np.array(losses), np.array(ephocs)

losses, ephocs = trainModel(dogs_vs_cats_predicter, training_data_loader, num_ephocs=3)

plt.plot(ephocs, losses, label="erro")

# Adicionando títulos e legendas
plt.title("Gráfico do erro")
plt.xlabel("x")
plt.ylabel("y")

# Exibir legenda
plt.legend()

# Exibindo o gráfico
plt.show()


torch.save(dogs_vs_cats_predicter.state_dict(), "dogs_vs_cats_model.pth")
