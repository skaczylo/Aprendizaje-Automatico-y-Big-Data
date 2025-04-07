from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch

"""
Clase Agente contiene la siguiente info:
- Dataset
- Modelo de la red que se quiera entrenar
- Numero de epoch
- Criterio y optimizador

Ademas contiene los metodos:

- entrenamientoTest = divide el dataset en entrenamiento y test
- entrenarModelo : entrena el modelo
"""


class Agente:

    def __init__(self,dataset,modelo,device, epoch, criterio, optimizador):

        #Datos principales
        self.dataset = dataset #Dataset
        self.modelo = modelo #Red neuronal
        self.device = device #CPU o GPU depende del ordenador
        
        #Metadatos
        self.epoch = epoch
        self.criterio = criterio
        self.optimizador = optimizador


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
    
    def entrenarModelo(self,train_data):
        for epoch in range(2):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(train_data, 0):
                # get the inputs; data is a list of [inputs, labels]
                imagenes, _, generos, _ = data
                imagenes, generos = imagenes.to(self.device),generos.to(self.device)

                # zero the parameter gradients
                self.optimizador.zero_grad()

                # forward + backward + optimize
                generosPred = self.modelo(imagenes)
        
                lossGenero = self.criterio(generosPred,generos)
       
                lossGenero.backward()
                self.optimizador.step()

                # print statistics  
                running_loss += lossGenero.item()
                if i % 1000 == 0:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 1000:.3f}')
                    running_loss = 0.0


        print('Finished Training')
    
