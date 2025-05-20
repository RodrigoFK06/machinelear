from fastapi import APIRouter, HTTPException
import pandas as pd
import os

router = APIRouter()

@router.get("/labels", response_model=list[str])
def get_labels():
    dataset_path = os.path.join("app", "legacy", "dataset_medico.csv")

    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail="Archivo de dataset no encontrado.")

    try:
        df = pd.read_csv(dataset_path, header=None)
        label_col = df.columns[-1]
        labels = df[label_col].dropna().unique().tolist()
        labels = sorted([label.strip() for label in labels])
        return labels
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error leyendo etiquetas: {str(e)}")
