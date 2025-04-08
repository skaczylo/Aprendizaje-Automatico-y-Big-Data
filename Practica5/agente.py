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
        self.modeloEdad= modeloEdad.to(device) #Red neuronal para la edad
        self.criterioEdad = criterioEdad
        self.optimizadorEdad = optim.Adam(self.modeloEdad.parameters(), lr=self.lr )

        #Modelo  genero
        self.modeloGenero= modeloGenero.to(device) #Red neuronal para la edad
        self.criterioGenero = criterioGenero
        self.optimizadorGenero = optim.Adam(self.modeloGenero.parameters(), lr=self.lr )

        #Modelo raza
        self.modeloRaza= modeloRaza.to(device) #Red neuronal para la edad
        self.criterioRaza = criterioRaza
        self.optimizadorRaza = optim.Adam(self.modeloRaza.parameters(), lr=self.lr )
        

    def entrenarEdad(self,train_data,num_epochs):

        for epoch in range(num_epochs):  # loop over the dataset multiple times

            running_loss = 0.0
         
            for i, batch in enumerate(train_data, 0):
                # get the inputs; data is a list of [inputs, labels]
                imagenes, edades, _, _ = batch
                imagenes, edades = imagenes.to(self.device),edades.to(self.device)

                # zero the parameter gradients
                self.optimizadorEdad.zero_grad()
               
                # forward + backward + optimize
                edadesPred = self.modeloEdad(imagenes)
             
                #Calculamos error
                lossEdad= self.criterioEdad(edadesPred.squeeze(),edades)

                #Edad
                lossEdad.backward()
                self.optimizadorEdad.step()

                # print statistics  
                running_loss += lossEdad.item()
             
            print(f'[{epoch + 1}, edad] loss: {running_loss / len(train_data)*4:.3f}')
        
        print('Finished Training')


    def entrenarGenero(self,train_data):

        for epoch in range(self.epochs):  # loop over the dataset multiple times

            running_loss = 0.0
         
            for i, batch in enumerate(train_data, 0):
                # get the inputs; data is a list of [inputs, labels]
                imagenes, _, generos, _ = batch
                imagenes, generos = imagenes.to(self.device),generos.to(self.device)

                # zero the parameter gradients
                self.optimizadorGenero.zero_grad()
               
                # forward + backward + optimize
                generosPred = self.modeloGenero(imagenes)
                #Calculamos error
                lossGenero = self.criterioGenero(generosPred,generos)
               
                #Genero
                lossGenero.backward()
                self.optimizadorGenero.step() 

                running_loss += lossGenero.item()
              
            print(f'[{epoch + 1},genero] loss: {running_loss / len(train_data)*4:.3f}')
    
        print('Finished Training')



    def entrenarRaza(self,train_data):

        for epoch in range(self.epochs):  # loop over the dataset multiple times

            running_loss = 0.0
         
            for i, batch in enumerate(train_data, 0):
                # get the inputs; data is a list of [inputs, labels]
                imagenes, _, _, razas = batch
                imagenes, razas = imagenes.to(self.device),razas.to(self.device)

                # zero the parameter gradients
                self.optimizadorRaza.zero_grad()
               

                # forward + backward + optimize
                razasPred = self.modeloRaza(imagenes)
                #Calculamos error
                lossRaza = self.criterioRaza(razasPred,razas)
               
                #Genero
                lossRaza.backward()
                self.optimizadorRaza.step() 

                running_loss += lossRaza.item()
              
            print(f'[{epoch + 1},raza] loss: {running_loss / len(train_data)*4:.3f}')
    
        print('Finished Training')
    
