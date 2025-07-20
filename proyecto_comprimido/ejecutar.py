#!/usr/bin/env python3
# Proyecto final - Alineación de transcripciones
# Este script ejecuta todo el proceso de alineación
import os
import sys
from transcription_alignment import AlineadorTranscripcion


def run_alignment():
    # Función principal que hace todo el trabajo

    print("=== Mi proyecto de alineación ===")
    print()

    # Configuración de rutas de archivos
    html_dir = "data/training/202003301830"
    textgrid_v = "data/training/202003301830_v.TextGrid"
    output_dir = "output"

    # Verificar que existan los archivos
    if not os.path.exists(html_dir):
        print(f"No existe el directorio {html_dir}")
        return False

    if not os.path.exists(textgrid_v):
        print(f"No existe el archivo {textgrid_v}")
        return False

    print(f"HTML: {html_dir}")
    print(f"TextGrid: {textgrid_v}")
    print(f"Output: {output_dir}")
    print()

    # Crear el objeto que hace la alineación
    aligner = AlineadorTranscripcion()

    try:
        print("Empezando...")
        result = aligner.procesar_alineacion(html_dir, textgrid_v, output_dir)

        print("\n" + "="*50)
        print("TERMINADO!")
        print("="*50)

        # Mostrar los resultados
        print(f"Segmentos creados: {len(result['segmentos_transc'])}")
        print(f"Segmentos IPU: {len(result['segmentos_ipu'])}")
        print(f"Errores: {len(result['segmentos_errores'])}")
        print(f"Similitud: {result['similitud_promedio']:.3f}")
        print(f"Errores %: {result['tasa_errores']:.3f}")

        print(f"\nArchivos creados:")
        for file in sorted(os.listdir(output_dir)):
            print(f"   {file}")

        print(f"\nQué hacer ahora:")
        print(f"1. Abrir Praat")
        print(f"2. Abrir el audio")
        print(f"3. Abrir el TextGrid: {output_dir}/transcripcion_alineada.TextGrid")
        print(f"4. Ver si la alineación está bien")
        print(f"5. Revisar errores")

        return True

    except Exception as e:
        print(f"\nError: {e}")
        return False


if __name__ == "__main__":
    # Si ejecuto el script directamente
    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help', 'help']:
            print("python ejecutar.py")
            print("Ejecuta mi programa de alineación")
        else:
            print("No existe esa opción")
    else:
        success = run_alignment()
        if not success:
            print("Algo salió mal :(")
        sys.exit(0 if success else 1)
