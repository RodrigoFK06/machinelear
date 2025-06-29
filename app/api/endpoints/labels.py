from fastapi import APIRouter, HTTPException
import pandas as pd
import os
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/labels",
            response_model=list[dict],
            summary="Obtener etiquetas de señas y sus niveles",
            description="Retorna una lista de señas médicas únicas con su nivel (principiante, intermedio o avanzado).")
def get_labels():
    dataset_path = os.path.join("D:", os.sep, "machinelear", "data", "dataset_medico.csv")

    if not os.path.exists(dataset_path):
        logger.error("Dataset file not found at %s", dataset_path)
        raise HTTPException(status_code=404, detail="Archivo de dataset no encontrado.")

    try:
        df = pd.read_csv(dataset_path, header=None)

        if df.empty or df.shape[1] < 2:
            raise HTTPException(status_code=500, detail="El dataset está vacío o no tiene suficientes columnas.")

        label_col = df.columns[-2]  # penúltima columna
        level_col = df.columns[-1]  # última columna

        df = df[[label_col, level_col]].dropna().drop_duplicates()

        # Validar niveles permitidos
        valid_levels = {"principiante", "intermedio", "avanzado"}
        df = df[df[level_col].str.lower().isin(valid_levels)]

        result = df.rename(columns={label_col: "label", level_col: "level"}) \
                   .sort_values(by=["level", "label"]) \
                   .to_dict(orient="records")

        return result

    except Exception as e:
        logger.exception("Error procesando el archivo del dataset.")
        raise HTTPException(status_code=500, detail=f"Error leyendo etiquetas y niveles: {str(e)}")