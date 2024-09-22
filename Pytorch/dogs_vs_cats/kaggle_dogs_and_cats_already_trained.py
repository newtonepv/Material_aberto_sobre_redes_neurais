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
dogs_vs_cats_test = "/home/edvsimoes/Downloads/dogs-vs-cats/test1"

# Transforms e DataLoader
transformations = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
])

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

test_dataset = datasets.ImageFolder(
    dogs_vs_cats_test,
    transform=transformations,
    loader=pil_loader
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

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

        self.dropout = torch.nn.Dropout(p=0.15)

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
    
dogs_vs_cats_predicter = Dogs_vs_cats_predicter_untrained()
state_dict = torch.load("dogs_vs_cats_model_trained2Times.pth")

# Carregar o state_dict no modelo
dogs_vs_cats_predicter.load_state_dict(state_dict)
dogs_vs_cats_predicter.eval()

# Teste do modelo
imagens, legendas = [], []
for i, (xs, _) in enumerate(test_loader):
    if i >= 20:  # Mostrando apenas 20 imagens
        break

    xs = xs.to(device)

    # Predição do modelo
    pred = dogs_vs_cats_predicter(xs).argmax(axis=1)

    # Armazena as imagens e as predições
    imagens.extend(xs.cpu())
    animal = "cachorro" if pred.item() == 1 else "gato"
    legendas.append(f"chute: {animal}")

# Crie uma grade de 4x5 (4 linhas e 5 colunas)
fig, axes = plt.subplots(4, 5, figsize=(15, 12))

# Itere sobre cada imagem e plote com a legenda correta
for i, (img, ax) in enumerate(zip(imagens, axes.flatten())):
    # Converta o tensor para numpy
    img_np = img.squeeze(0).permute(1, 2, 0).numpy()  # Troca a ordem dos canais de [C, H, W] para [H, W, C]
    
    # Desnormaliza para o intervalo [0, 1]
    img_np = 0.5 * img_np + 0.5  # Reverte a normalização

    # Mostra a imagem
    ax.imshow(img_np)
    
    # Adiciona legenda
    ax.set_title(legendas[i])
    
    # Remove os eixos
    ax.axis('off')

# Ajusta os espaçamentos
plt.tight_layout()
plt.show()
