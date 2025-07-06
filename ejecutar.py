#!/usr/bin/env python3

import os
import sys
import subprocess

def run_alignment():
    """Ejecutar el sistema de alineación"""

    data_dir = "data/training/202003301830"
    output_dir = "output"

    # ver si están los archivos
    textgrid_file = os.path.join(data_dir, "202003301830.TextGrid")
    audio_file = os.path.join(data_dir, "202003301830.mp3")

    if not os.path.exists(textgrid_file):
        print(f"Error: No se encuentra {textgrid_file}")
        return False

    if not os.path.exists(audio_file):
        print(f"Error: No se encuentra {audio_file}")
        return False

    # crear carpeta de salida
    os.makedirs(output_dir, exist_ok=True)

    # ejecutar con todos los HTML
    cmd = [
        sys.executable,
        "transcription_alignment.py",
        data_dir,
        textgrid_file,
        audio_file,
        output_dir
    ]

    print("Ejecutando alineación...")
    print(f"Comando: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True)
        print("Proceso completado")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    if run_alignment():
        print("\nArchivos en output/:")
        for f in os.listdir("output"):
            print(f"  - {f}")
    else:
        print("\nError, algo salió mal")
        sys.exit(1)
