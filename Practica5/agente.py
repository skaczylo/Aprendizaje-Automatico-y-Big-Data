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
        
        
    def entrenarModelo(self, train_data, tarea, num_epochs):

        #Mapeamos el modelo, optimizador y criterio segun la tarea
        modelo = getattr(self, f'modelo{tarea.capitalize()}')
        optimizador = getattr(self, f'optimizador{tarea.capitalize()}')
        criterio = getattr(self, f'criterio{tarea.capitalize()}')

        # Selector de etiquetas según la tarea
        idx_etiqueta = {"Edad": 1, "Genero": 2, "Raza": 3}[tarea.capitalize()]

        for epoch in range(num_epochs):
            running_loss = 0.0

            for batch in train_data:
                imagenes = batch[0].to(self.device)
                etiquetas = batch[idx_etiqueta].to(self.device)

                optimizador.zero_grad()
                predicciones = modelo(imagenes)

                if tarea.lower() == "edad":
                    predicciones = predicciones.squeeze()

                loss = criterio(predicciones, etiquetas)
                loss.backward()
                optimizador.step()

                running_loss += loss.item()
            print(f'[{epoch + 1}, edad] loss: {running_loss / len(train_data)*4:.3f}')
 

        print(f"Finished Training {tarea}")


    def resultados(self,datos):
        generosTotal = torch.tensor([]).to(self.device)
        generosPredTotal = torch.tensor([]).to(self.device)

        edadesTotal = torch.tensor([]).to(self.device)
        edadesPredTotal = torch.tensor([]).to(self.device)

        razasTotal = torch.tensor([]).to(self.device)
        razasPredTotal = torch.tensor([]).to(self.device)

        with torch.no_grad():
            for batch in datos:

                imagenes,edades,generos,razas = batch
                imagenes, edades, generos,razas = imagenes.to(self.device),edades.to(self.device),generos.to(self.device),razas.to(self.device)

                #Genero
                generosPred = self.modeloGenero(imagenes)
                _, generosPred = torch.max(generosPred, 1)

                generosPredTotal =torch.cat((generosPredTotal,generosPred),dim = 0)
                generosTotal = torch.cat((generosTotal,generos),dim = 0)

                #Edad
                edadesPred = self.modeloEdad(imagenes)
                edadesPredTotal = torch.cat((edadesPredTotal,edadesPred.squeeze().int()),dim = 0)
                edadesTotal = torch.cat((edadesTotal,edades), dim = 0)


                #Raza
                razasPred = self.modeloRaza(imagenes)
                _, razasPred = torch.max(razasPred, 1)
        
                razasPredTotal =torch.cat((razasPredTotal,razasPred),dim = 0)
                razasTotal = torch.cat((razasTotal,razas),dim = 0)

        return edadesTotal,edadesPredTotal,generosTotal,generosPredTotal,razasTotal,razasPredTotal
    
