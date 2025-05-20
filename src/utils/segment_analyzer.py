import pandas as pd
import numpy as np

FRAMES = 35
FEATURES = 42

def cargar_referencias(path_csv, nombre_sena):
    df = pd.read_csv(path_csv, header=None)
    etiqueta_col = df.columns[-1]
    df = df[df[etiqueta_col] == nombre_sena]

    if df.empty:
        return None

    secuencias = df.iloc[:, :-1].values
    return secuencias.reshape((-1, FRAMES, FEATURES))

def dividir_en_tres(secuencia):
    return [
        secuencia[0:12],
        secuencia[12:24],
        secuencia[24:35]
    ]

def promedio_por_segmento(secuencias):
    partes = [[], [], []]  # inicio, mitad, final
    for seq in secuencias:
        inicio, mitad, final = dividir_en_tres(seq)
        partes[0].append(inicio)
        partes[1].append(mitad)
        partes[2].append(final)
    return [np.mean(p, axis=0) for p in partes]

def distancia_segmento(secuencia_usuario, referencias):
    resultados = []
    partes_usuario = dividir_en_tres(np.array(secuencia_usuario))
    for i, ref in enumerate(referencias):
        dist = np.linalg.norm(partes_usuario[i] - ref)
        resultados.append(dist)
    return resultados

def analizar_error(secuencia_usuario, dataset_csv, nombre_sena):
    ref_data = cargar_referencias(dataset_csv, nombre_sena)
    if ref_data is None:
        return "No hay datos de referencia para esa seña."

    refs = promedio_por_segmento(ref_data)
    dists = distancia_segmento(secuencia_usuario, refs)
    secciones = ["inicial", "media", "final"]
    indice = np.argmax(dists)
    return f"Observación: El error ocurrió principalmente en la parte {secciones[indice]}."
