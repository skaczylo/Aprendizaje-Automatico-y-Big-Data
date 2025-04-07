from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
import torch.optim as optim

"""
Esta clase contiene 3 modelos uno para cada atributo y entrena los 3 modelos a la vez para optimizar el tiempo
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

    def __init__(self,modeloEdad,modeloGenero, modeloRaza, criterioEdad, criterioGenero, criterioRaza, device, epochs,lr):

        #Datos principales
        self.device = device #CPU o GPU depende del ordenador
        self.epochs = epochs
        self.lr = lr

        #Modelo edad
        self.modeloEdad= modeloEdad #Red neuronal para la edad
        self.criterioEdad = criterioEdad
        self.optimizadorEdad = optim.Adam(self.modeloEdad.parameters(), lr=self.lr )

        #Modelo  genero
        self.modeloGenero= modeloGenero #Red neuronal para la edad
        self.criterioGenero = criterioGenero
        self.optimizadorGenero = optim.Adam(self.modeloGenero.parameters(), lr=self.lr )

        #Modelo raza
        self.modeloRaza= modeloRaza #Red neuronal para la edad
        self.criterioRaza = criterioRaza
        self.optimizadorRaza = optim.Adam(self.modeloRaza.parameters(), lr=self.lr )
        

    def entrenarAgente(self,train_data):

        for epoch in range(self.epochs):  # loop over the dataset multiple times

            running_loss_edad = 0.0
            running_loss_genero = 0.0
            running_loss_raza = 0.0

            for i, batch in enumerate(train_data, 0):
                # get the inputs; data is a list of [inputs, labels]
                imagenes, edades, generos, razas = batch
                imagenes, edades, generos, razas = imagenes.to(self.device),edades.to(self.device),generos.to(self.device),razas.to(self.device)

                # zero the parameter gradients
                self.optimizadorEdad.zero_grad()
                self.optimizadorEdad.zero_grad()
                self.optimizadorEdad.zero_grad()

                # forward + backward + optimize
                edadesPred = self.modeloEdad(imagenes)
                generosPred = self.modeloGenero(imagenes)
                razasPred = self.modeloRaza(imagenes)
        
                #Calculamos error
                lossEdad= self.criterioEdad(edadesPred.squeeze(),edades)
                lossGenero = self.criterioGenero(generosPred,generos)
                lossRaza = self.criterioRaza(razasPred,razas)
       
                #Edad
                lossEdad.backward()
                self.optimizadorEdad.step()

                #Genero
                lossGenero.backward()
                self.optimizadorGenero.step()

                #Raza
                lossRaza.backward()
                self.optimizadorRaza.step()

                # print statistics  
                running_loss_edad += lossEdad.item()
                running_loss_genero += lossGenero.item()
                running_loss_raza += lossRaza.item()


            print(f'[{epoch + 1}, edad] loss: {running_loss_edad / len(train_data)*4:.3f}')
            print(f'[{epoch + 1}, genero] loss: {running_loss_genero / len(train_data)*4:.3f}')
            print(f'[{epoch + 1}, raza] loss: {running_loss_raza / len(train_data)*4:.3f}')
    
        print('Finished Training')
    
