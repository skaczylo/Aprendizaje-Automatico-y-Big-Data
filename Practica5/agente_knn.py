
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import root_mean_squared_error
from data import ImgDataset
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import random
import torch
from sklearn.model_selection import cross_val_score
from lime import lime_image
from skimage.segmentation import mark_boundaries
import lime 
class ImgDatasetAux(ImgDataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        super().__init__(img_dir, transform, target_transform)  # Llamar al constructor de la clase base

    def __getitem__(self, idx):
        # Obtener los datos originales de la clase base
        imagen, edad, genero, raza = super().__getitem__(idx)
        
        # Clasificar la edad utilizando la función clasificar_edad
        edad_clasificada = clasificar_edad(edad.item())  # Convertir el tensor a número y clasificar

        # Devolver la imagen y las etiquetas modificadas
        return imagen, edad_clasificada, genero, raza
    
def clasificar_edad(edad):
    """
    Clasifica la edad en uno de los 10 grupos de acuerdo con el rango de edad.
    """
    if 0 <= edad <= 9:
        return 0
    elif 10 <= edad <= 19:
        return 1
    elif 20 <= edad <= 29:
        return 2
    elif 30 <= edad <= 39:
        return 3
    elif 40 <= edad <= 49:
        return 4
    elif 50 <= edad <= 59:
        return 5
    elif 60 <= edad <= 69:
        return 6
    elif 70 <= edad <= 79:
        return 7
    elif 80 <= edad <= 89:
        return 8
    elif 90 <= edad:
        return 9
    else:
        raise ValueError("Edad fuera de los límites permitidos (0-100)")
    

def entrenar_knn(dataloader, etiqueta, n_neighbors):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    if etiqueta not in ["raza", "edad", "genero"]:
        raise ValueError("Etiqueta debe ser 'raza', 'edad' o 'genero'")
    
    X_total = []
    y_total = []

    for imagenes, edades, generos, razas in dataloader:
        imagenes_flat = imagenes.view(imagenes.size(0), -1).cpu().numpy()
        
        if etiqueta == "raza":
            etiquetas = razas.cpu().numpy()
        elif etiqueta == "edad":
            etiquetas = edades.cpu().numpy()
        elif etiqueta == "genero":
            etiquetas = generos.cpu().numpy()
        
        X_total.append(imagenes_flat)
        y_total.append(etiquetas)

    # Concatenar todos los mini-lotes
    X_total = np.concatenate(X_total, axis=0)
    y_total = np.concatenate(y_total, axis=0)

    # Entrenar una única vez con todo
    knn.fit(X_total, y_total)

    return knn

def extraer_subconjunto(dataloader, num_imagenes=500, categoria="raza"):
    """
    Extrae un subconjunto de imágenes y sus etiquetas de una categoría específica.
    
    Args:
    dataloader (DataLoader): El DataLoader con el conjunto de datos.
    num_imagenes (int): El número máximo de imágenes a extraer.
    categoria (str): La categoría para la cual se deben extraer las etiquetas ('raza', 'edad', 'genero').
    
    Returns:
    tuple: Un conjunto de imágenes y sus etiquetas correspondientes a la categoría.
    """
    X_train_subset = []  # Para almacenar las características (imágenes aplanadas)
    y_train_subset = []  # Para almacenar las etiquetas correspondientes a la categoría

    # Iteramos sobre los mini-lotes en el DataLoader
    for imagenes, edades, generos, razas in dataloader:
        if len(X_train_subset) >= num_imagenes:  # Si ya tenemos suficientes imágenes, salimos
            break

        # Aplanamos las imágenes de tamaño (batch_size, 3, 200, 200) a (batch_size, 120000)
        imagenes_flat = imagenes.view(imagenes.size(0), -1).cpu().numpy()

        # Seleccionamos las etiquetas según la categoría
        if categoria == "raza":
            etiquetas = razas.cpu().numpy()
        elif categoria == "edad":
            etiquetas = edades.cpu().numpy()
        elif categoria == "genero":
            etiquetas = generos.cpu().numpy()

        # Añadir las imágenes y las etiquetas al subconjunto
        X_train_subset.append(imagenes_flat)
        y_train_subset.append(etiquetas)

    # Concatenamos todas las imágenes y las etiquetas
    X_train_subset = np.concatenate(X_train_subset, axis=0)
    y_train_subset = np.concatenate(y_train_subset, axis=0)

    return X_train_subset, y_train_subset

def establecer_semilla(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def encontrar_mejor_k(dataloader, categoria, num_imagenes=500, k_range=range(1, 21)):

    establecer_semilla()
    # Aseguramos que la categoría sea válida
    if categoria not in ["raza", "edad", "genero"]:
        raise ValueError("La categoría debe ser 'raza', 'edad' o 'genero'")
    
    # Extraer un subconjunto de datos (limitado a 'num_imagenes' imágenes)
    X_train_subset, y_train_subset = extraer_subconjunto(dataloader, num_imagenes=num_imagenes, categoria=categoria)

    # Almacenar las puntuaciones para cada valor de k
    mejores_scores = []

    # Probar diferentes valores de k
    for k in k_range:
        # Configurar el clasificador k-NN
        knn = KNeighborsClassifier(n_neighbors=k)
        
        # Realizar la validación cruzada usando cross_val_score
        scores = cross_val_score(knn, X_train_subset, y_train_subset, cv=5)  # cv=5 es 5 pliegues
        
        # Almacenar la media de la puntuación de cada valor de k
        mejores_scores.append(scores.mean())

        # Imprimir la puntuación para este valor de k
        print(f"k={k} -> Puntuación media de la validación cruzada: {scores.mean()}")

    # Encontrar el valor de k que produce la mejor puntuación media
    mejor_k = k_range[np.argmax(mejores_scores)]
    mejor_score = max(mejores_scores)

    print(f"\nEl valor óptimo de k es: {mejor_k} con una puntuación media de: {mejor_score}")

    # Graficar los resultados
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, mejores_scores, marker='o', linestyle='-', color='b', label='Puntuación media CV')
    plt.title(f"Puntuación media de la validación cruzada para diferentes valores de k ({categoria})")
    plt.xlabel('k')
    plt.ylabel('Puntuación media')
    plt.grid(True)
    plt.xticks(k_range)
    plt.legend()
    plt.show()

    return mejor_k, mejor_score

from sklearn.metrics import accuracy_score, mean_squared_error, classification_report

def evaluar_modelo(modelo, dataloader, etiqueta="genero"):
    """
    Evalúa un modelo de sklearn usando un DataLoader de PyTorch.
    Calcula precisión (accuracy), RMSE y muestra precision/recall/F1-score.

    Args:
        modelo: Modelo de sklearn ya entrenado (por ejemplo, KNeighborsClassifier).
        dataloader: DataLoader que entrega (imagenes, edades, generos, razas).
        etiqueta: Qué etiqueta usar para evaluación ('raza', 'edad' o 'genero').

    Returns:
        accuracy: Precisión del modelo en el conjunto de prueba.
        rmse: Raíz del error cuadrático medio en el conjunto de prueba.
        report: Reporte de clasificación (precision, recall, f1-score, support).
        y_test: Etiquetas reales.
        y_pred: Etiquetas predichas por el modelo.
    """
    X_test = []
    y_test = []

    for imagenes, edades, generos, razas in dataloader:
        imagenes_flat = imagenes.view(imagenes.size(0), -1).cpu().numpy()

        if etiqueta == "raza":
            etiquetas = razas.cpu().numpy()
        elif etiqueta == "edad":
            etiquetas = edades.cpu().numpy()
        elif etiqueta == "genero":
            etiquetas = generos.cpu().numpy()
        else:
            raise ValueError("Etiqueta debe ser 'raza', 'edad' o 'genero'")

        X_test.append(imagenes_flat)
        y_test.append(etiquetas)

    # Concatenar todos los lotes
    X_test = np.concatenate(X_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    # Predicciones del modelo
    y_pred = modelo.predict(X_test)

    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    report = classification_report(y_test, y_pred, digits=4)

    print(f"Precisión (accuracy) en el conjunto de prueba: {accuracy:.4f}")
    print(f"Error cuadrático medio (RMSE) en el conjunto de prueba: {rmse:.4f}")
    print("\nReporte de clasificación:")
    print(report)

    return y_test, y_pred

def lime_explain_and_visualize(modelo, imagen, device='cpu', num_samples=1000):
    """
    Genera una explicación de LIME para una imagen usando un modelo y visualiza el resultado.
    
    Args:
        modelo: El modelo entrenado que queremos explicar (por ejemplo, un modelo k-NN, CNN, etc.).
        imagen: La imagen a la que queremos generar la explicación (tensor de tamaño (C, H, W)).
        device: El dispositivo en el que se encuentra el modelo (CPU o GPU).
        num_samples: Número de muestras que LIME generará para la explicación.
    """
    
    # Obtener la explicación de LIME
    explanation = lime_explanation(modelo, imagen, device, num_samples)
    
    # Verificar las etiquetas disponibles en la explicación
    print("Etiquetas disponibles en la explicación:", explanation.local_exp.keys())
    
    # Elegir la etiqueta más probable (la primera de la explicación)
    label = list(explanation.local_exp.keys())[0]
    
    # Obtener la imagen explicada y su máscara
    temp, mask = explanation.get_image_and_mask(label, positive_only=True, num_features=10, hide_rest=False)
    
    # Mostrar la imagen original y la explicación
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    # Convertir la imagen de tensor a formato numpy (H, W, C) para visualizarla
    imagen_np = imagen.permute(1, 2, 0).cpu().numpy()  # Convertir (C, H, W) a (H, W, C)
    
    # Mostrar la imagen original (sin desnormalizar si se quiere mostrar como estaba originalmente)
    ax[0].imshow(imagen_np)  # Imagen original
    ax[0].set_title("Imagen Original")
    ax[0].axis('off')
    
    # Mostrar la explicación LIME
    ax[1].imshow(temp)
    ax[1].imshow(mask, cmap='jet', alpha=0.5)  # Superponer la máscara sobre la imagen
    ax[1].set_title("Explicación LIME")
    ax[1].axis('off')
    
    plt.show()

def lime_explanation(modelo, imagen, device, num_samples=1000):
    """
    Crea una explicación utilizando LIME para una imagen y un modelo.
    
    Args:
        modelo: El modelo entrenado que queremos explicar (por ejemplo, un modelo k-NN, CNN, etc.).
        imagen: La imagen a la que queremos generar la explicación (tensor de tamaño (C, H, W)).
        device: El dispositivo en el que se encuentra el modelo (CPU o GPU).
        num_samples: Número de muestras que LIME generará para la explicación.
    
    Returns:
        explanation: Explicación generada por LIME.
    """
    
    # Asegúrate de que la imagen esté en el formato adecuado (batch_size, C, H, W)
    imagen = imagen.unsqueeze(0).to(device)  # Añadir la dimensión de batch (1, C, H, W)
    
    # Preprocesamiento de la imagen (debe ser el mismo que usaste para el entrenamiento del modelo)
    imagen_np = imagen.squeeze().cpu().numpy().transpose(1, 2, 0)  # Convertir de (C, H, W) a (H, W, C)
    
    # Función que hace la predicción para una imagen (utilizada por LIME)
    def predict_fn(images):
        # Convertir las imágenes a formato tensor y luego a numpy
        images_tensor = torch.tensor(images).permute(0, 3, 1, 2).float().to(device)  # (batch_size, C, H, W)
        images_tensor = images_tensor.cpu().numpy()  # Convertir a numpy para KNN
        
        # Aplanar las imágenes si es necesario para el modelo KNN (debe ser igual al preprocesamiento del entrenamiento)
        images_flat = images_tensor.reshape(images_tensor.shape[0], -1)  # Aplanar las imágenes (batch_size, C*H*W)

        # Usar `predict_proba` si es KNN o cualquier modelo que devuelva probabilidades
        return modelo.predict_proba(images_flat)  # Predecir probabilidades de clasificación

    # Crear el explicador de LIME
    explainer = lime_image.LimeImageExplainer()

    # Generar la explicación con LIME
    explanation = explainer.explain_instance(imagen_np, predict_fn, top_labels=1, num_samples=num_samples)

    return explanation

def predecir_instancia(modelo, imagen, etiqueta="genero"):
    """
    Evalúa un modelo de sklearn usando una única instancia de imagen y devuelve la predicción.

    Args:
        modelo: Modelo de sklearn ya entrenado (por ejemplo, KNeighborsClassifier).
        imagen: Instancia de imagen a evaluar (tensor de tamaño (C, H, W)).
        etiqueta: Qué etiqueta usar para evaluación ('raza', 'edad' o 'genero').

    Returns:
        prediccion: La predicción de la clase para la instancia dada.
    """
    # Asegurarse de que la imagen esté en el formato adecuado (batch_size, C, H, W)
    imagen = imagen.unsqueeze(0)  # Añadir dimensión de batch (1, C, H, W)

    # Aplanar la imagen (convertirla en un vector)
    imagen_flat = imagen.view(imagen.size(0), -1).cpu().numpy()

    # Predecir usando el modelo
    prediccion = modelo.predict(imagen_flat)  # Predicción de la clase

    return prediccion[0]  # Regresamos la predicción para la instancia

def predecir_varias_instancias(modelo, imagenes, etiqueta="genero"):
    """
    Evalúa un modelo de sklearn usando un lote de imágenes y devuelve las predicciones.

    Args:
        modelo: Modelo de sklearn ya entrenado (por ejemplo, KNeighborsClassifier).
        imagenes: Lote de imágenes a evaluar (tensor de tamaño (batch_size, C, H, W)).
        etiqueta: Qué etiqueta usar para evaluación ('raza', 'edad' o 'genero').

    Returns:
        predicciones: Las predicciones de las clases para el lote de imágenes.
    """
    # Asegurarse de que las imágenes estén en el formato adecuado (batch_size, C, H, W)
    imagenes_flat = imagenes.view(imagenes.size(0), -1).cpu().numpy()  # Aplanar todas las imágenes

    # Predecir usando el modelo (devuelve un arreglo de predicciones para todas las imágenes)
    predicciones = modelo.predict(imagenes_flat)

    return predicciones  # Regresamos las predicciones para todas las instancias en el batch


