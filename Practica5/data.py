#Vamos a trabajar con pytorch y este permite crear tu propio dataset
from torch.utils.data import Dataset
import os
from PIL import Image
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt
import numpy as np




"""
Esta clase sirve para tratar las imagenes de manera mas optima con la biblioteca pytorch
Es una clase que hereda de Dataset (perteneciente a pytorch)

Estructura de las imagenes:
- 200x200 pixeles
- Nombre imagen : edad_genero_raza_fecha.jpg
- edad: cualquier numero entre 1 y 100
- genero : 0 = hombre y 1= mujer
- raza : 5 clases

__getitem__ se encarga de leer el nombre de la foto y separar los datos; los convierte en tensores y si hay
algun transformador de los datos los transforma
Devuelve la imagen, el año, el genero y la raza

"""

class ImgDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_labels = os.listdir(img_dir) #lista de los nombres de las fotos
        self.img_dir = img_dir #Ruta principal donde estan las imagenes
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):

        nombre_imagen = os.path.join(self.img_dir, self.img_labels[idx])  #Ruta completa
        imagen =  Image.open(nombre_imagen) #abrimos la imagen

        # Extraer la info del nombre del archivo
        parts = self.img_labels[idx].split('_')  # Separar por guiones bajos

        age = torch.tensor(int(parts[0])).float()  # Primer elemento: la edad
        gender = torch.tensor(int(parts[1]))  # Segundo elemento: el género
        race = torch.tensor(int(parts[2]))
        
        if self.transform:

          imagen = self.transform(imagen)  # Aplicar transformaciones a la imagen

        if self.target_transform:
          etiqueta = self.target_transform(etiqueta)  # Aplicar transformaciones a la etiqueta (opcional)

        return imagen, age,gender,race

"""
funcion que muestra las imagenes
""" 
def mostrarImg(img):
    img = torchvision.utils.make_grid(img)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


"""
entrenamientoTest se encarga de dividir el dataset total en conjunto de entrenamiento y test
"""

def entrenamientoTest(dataset,test_size = 0.2,batch_size = 4):
       
    #Primero realizamos una division de  indices
    train_indices, test_indices = train_test_split(range(len(dataset)), test_size=test_size, random_state=42)

    #Dividimos los indices
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    test_subset = torch.utils.data.Subset(dataset, test_indices)

    # Crear los DataLoaders con los indices que hemos repartidos
    train_dataloader = DataLoader(train_subset, batch_size=4, shuffle=True)
    test_dataloader = DataLoader(test_subset, batch_size=4, shuffle=False)

    #Dividimos ambos conjuntos en batches
    train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    return train_dataloader,test_dataloader
       