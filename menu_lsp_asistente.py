import os
import subprocess
import platform

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
UTILS_DIR = os.path.join(SRC_DIR, "utils")
PYTHON_EXE = os.path.join(BASE_DIR, ".venv", "Scripts", "python.exe") if platform.system() == "Windows" else "python3"

# Opciones visibles
OPTIONS = {
    "1": "Grabar seña estática (letra o palabra)",
    "2": "Grabar secuencia de frase (movimiento)",
    "3": "Predecir seña estática",
    "4": "Predecir frase en movimiento",
    "0": "Salir"
}

# Mapeo de opciones a scripts reales
SCRIPTS = {
    "1": os.path.join(BASE_DIR, "main.py"),
    "2": os.path.join(SRC_DIR, "utils", "grabar_secuencia_lstm.py"),
    "3": os.path.join(BASE_DIR, "realtime_predictor.py"),
    "4": os.path.join(BASE_DIR, "realtime_lstm_predictor.py"),
}

def show_menu():
    print("\n📋 MENÚ LSP ASISTENTE")
    for k in sorted(OPTIONS.keys()):
        print(f"{k}. {OPTIONS[k]}")

def ejecutar_opcion(script_path):
    if not os.path.exists(script_path):
        print(f"❌ El archivo no existe: {script_path}")
        return
    try:
        subprocess.run([PYTHON_EXE, script_path])
    except Exception as e:
        print(f"❌ Error al ejecutar: {e}")

def main():
    while True:
        show_menu()
        opcion = input("\nSelecciona una opción: ").strip()

        if opcion == "0":
            print("👋 Saliendo del asistente. ¡Hasta pronto!")
            break
        elif opcion in SCRIPTS:
            print(f"\n🚀 Ejecutando: {OPTIONS[opcion]}")
            ejecutar_opcion(SCRIPTS[opcion])
        else:
            print("⚠️ Opción inválida. Intenta nuevamente.")

if __name__ == "__main__":
    main()
