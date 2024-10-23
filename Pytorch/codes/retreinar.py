'''
mapas mentais relacionados:
https://lucid.app/lucidspark/1f8b320c-8114-4349-af78-05ec1cfd20be/edit?invitationId=inv_42e603f7-68e5-4d2d-86a3-e8d3326ae8b9&page=0_0
https://lucid.app/lucidspark/f3e8ec48-df8b-4da1-be7c-dd8e545b22f0/edit?page=0_0&invitationId=inv_84dd43d5-dbbd-4d86-b0c4-aa655b29adef#
'''
import torch #link do lucidchart
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

# Definições do dispositivo e dos caminhos das pastas (supondo que já estejam definidos)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dogs_train_folder_path = "../dogs-vs-cats/train/dogs"
cats_train_folder_path = "../dogs-vs-cats/train/cats"
database_train_folder = "../dogs-vs-cats/train"

# Transforms e DataLoader
transformations = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) #normaliza todos os itens do vetor dentre eles
])# é uma classe que armazena as transformações para modificar uma imagem


training_data_sets = datasets.ImageFolder(
    database_train_folder,
    transform=transformations,
)

training_data_loader = DataLoader(training_data_sets, batch_size=32, shuffle=True, num_workers=4)

# Definição do modelo
class Dogs_vs_cats_predicter_untrained(torch.nn.Module):
    def __init__(self):#nessa parte são definidas variáveis para representar cada camada, e função
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
        return x #é definido como a variável passa pelas camadas até se tornar a saída
    
dogs_vs_cats_predicter = Dogs_vs_cats_predicter_untrained()#cria uma instância do modelo

state = torch.load("dogs_vs_cats_model_trained2Times.pth", weights_only=True)#pega o modelo treinad dos arquivos

dogs_vs_cats_predicter.load_state_dict(state)#salva o modelo treinado na instância

dogs_vs_cats_predicter.train()#coloca ele no modo de treino, para calcular as derivadas

def trainModel(f, dl, num_ephocs=1):
    dogs_vs_cats_predicter.to(device)#usa cuda se necessário

    optim = torch.optim.SGD(f.parameters(), lr = 0.007)#define como vai treinar os pesos, nesse caso usa backpropagation

    error = torch.nn.CrossEntropyLoss()#define a função de erro, nesse casso cross entropy loss



    losses= [] #aqui serão salvos os erros
    ephocs= [] #aqui será salvo o eixo x (unidades de treino)
    N = len(dl)
    print(N)

    for e in range(num_ephocs):#por cada época (ephoc)
        data = iter(dl) #pega uma parte do data loader
        print(f"Ephoc: {e}",flush=True)
        for i in range(500): #roda por 500 imagens da parte do data loader

            x, y = next(data)
            x, y = x.to(device), y.to(device)#pega a entrada e a saida esperada

            print(i, end=" ",flush=True) #printa a imagem na epoca

            optim.zero_grad()#usa isso para calcular a derivada
            loss = error(f(x),y) #calcula o erro
            loss.backward() #calcula as derivadas
            optim.step() #ajusta os pesos em função das derivadas

            ephocs.append(e+i/N)
            losses.append(loss.item())
            #registra nas listas

    return np.array(losses), np.array(ephocs)

losses, ephocs = trainModel(dogs_vs_cats_predicter, training_data_loader, num_ephocs=3)#chama a função

#plota o grafico
plt.plot(ephocs, losses, label="erro")

# Adicionando títulos e legendas
plt.title("Gráfico do erro")
plt.xlabel("x")
plt.ylabel("y")

# Exibir legenda
plt.legend()

# Exibindo o gráfico
plt.show()

#guarda o modelo
torch.save(dogs_vs_cats_predicter.state_dict(), "dogs_vs_cats_model_trained2Times.pth")
