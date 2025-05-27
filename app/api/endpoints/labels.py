from fastapi import APIRouter, HTTPException
import pandas as pd
import os
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

# TODO: For improved scalability and performance with a large number of labels
# or high traffic, consider loading these labels from a database collection
# instead of a CSV file on each call.

@router.get("/labels",
            response_model=list[str],
            summary="Get all unique medical sign labels",
            description="Retrieves a sorted list of all unique medical sign labels available in the system, loaded from the medical dataset."
            )
def get_labels():
    dataset_path = os.path.join("app", "legacy", "dataset_medico.csv")

    if not os.path.exists(dataset_path):
        logger.error("Dataset file not found at %s", dataset_path)
        raise HTTPException(status_code=404, detail="Archivo de dataset no encontrado.")

    try:
        df = pd.read_csv(dataset_path, header=None)
        if df.empty:
            logger.warning("Dataset file at %s is empty.", dataset_path)
            return []

        label_col_index = -1 # Assuming label is the last column
        if df.shape[1] <= abs(label_col_index):
             logger.error("Dataset file %s does not have enough columns to find the label column.", dataset_path)
             raise HTTPException(status_code=500, detail="Formato de dataset incorrecto: no se encontró la columna de etiquetas.")

        label_col = df.columns[label_col_index]
        labels = df[label_col].dropna().unique().tolist()
        labels = sorted([str(label).strip() for label in labels]) # Ensure labels are strings
        return labels
    except pd.errors.EmptyDataError:
        logger.warning("Dataset file at %s is empty or could not be parsed by pandas.", dataset_path)
        raise HTTPException(status_code=500, detail="Error procesando el dataset: archivo vacío o malformado.")
    except Exception as e:
        logger.exception("Error processing dataset file %s", dataset_path)
        raise HTTPException(status_code=500, detail=f"Error leyendo etiquetas: {str(e)}")
