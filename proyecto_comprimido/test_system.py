#!/usr/bin/env python3
"""
Script de prueba para el sistema de alineación de transcripciones
"""
import os
import sys
from transcription_alignment import TranscriptionAligner

def test_training_data():
    """Probar con los datos de entrenamiento"""
    training_dir = "data/training/202003301830"
    html_file = os.path.join(training_dir, "01.html")
    textgrid_file = os.path.join(training_dir, "202003301830.TextGrid")
    audio_file = os.path.join(training_dir, "202003301830.mp3")
    output_dir = "output/test_training"

    # Verificar si están los archivos
    if not os.path.exists(html_file):
        print(f"Error: No se encontró el archivo HTML: {html_file}")
        return False

    if not os.path.exists(textgrid_file):
        print(f"Error: No se encontró el archivo TextGrid: {textgrid_file}")
        return False

    if not os.path.exists(audio_file):
        print(f"Error: No se encontró el archivo de audio: {audio_file}")
        return False

    print("=== PROBANDO EL SISTEMA ===")
    print(f"Archivo HTML: {html_file}")
    print(f"Archivo TextGrid: {textgrid_file}")
    print(f"Archivo de audio: {audio_file}")
    print(f"Directorio de salida: {output_dir}")
    print()

    # crear carpeta de salida
    os.makedirs(output_dir, exist_ok=True)

    # crear el alineador
    aligner = TranscriptionAligner()

    # probar cada parte
    print("1. Analizando HTML...")
    html_text = aligner.parse_transcription_html(html_file)
    print(f"   Longitud del texto extraído: {len(html_text)} caracteres")
    print(f"   Primeros 200 caracteres: {html_text[:200]}...")
    print()

    print("2. Evaluando IPU...")
    ipu_segments = aligner.extract_ipu_segments(textgrid_file)
    print(f"   Extraídos {len(ipu_segments)} segmentos IPU")
    if ipu_segments:
        print(f"   Primer segmento: {ipu_segments[0].start_time:.2f}s - {ipu_segments[0].end_time:.2f}s: '{ipu_segments[0].text[:50]}...'")
    print()

    print("3. Realizando alineación...")
    alignment_result = aligner.align_transcription_to_ipu(html_text, ipu_segments)
    print(f"   Creados {len(alignment_result.transcription_segments)} segmentos de transcripción")
    print(f"   Encontrados {len(alignment_result.errors)} errores potenciales")
    print()

    print("4. Generando TextGrid...")
    transc_output = os.path.join(output_dir, "test_transc.TextGrid")
    if aligner.generate_textgrid_transc(alignment_result, transc_output):
        print(f"   TextGrid guardado en: {transc_output}")
    else:
        print("   Error generando TextGrid")
    print()

    print("5. Buscando errores...")
    errors = aligner.detect_errors_with_espeak(
        alignment_result.transcription_segments[:5],  # Test first 5 segments
        alignment_result.ipu_segments[:5]
    )
    print(f"   Detectados {len(errors)} errores en los primeros 5 segmentos")
    print()

    print("6. Generando TextGrid de errores...")
    errors_output = os.path.join(output_dir, "test_errors.TextGrid")
    if aligner.generate_textgrid_transc_errors(errors, errors_output):
        print(f"   TextGrid de errores guardado en: {errors_output}")
    else:
        print("   Error generando TextGrid de errores")
    print()

    print("7. Haciendo reporte...")
    validation_report = aligner.play_and_validate(audio_file, errors)
    report_output = os.path.join(output_dir, "test_validation_report.txt")
    with open(report_output, 'w', encoding='utf-8') as f:
        f.write(validation_report)
    print(f"   Reporte de validación guardado en: {report_output}")
    print()

    print("=== PRUEBA FINAL ===")
    results = aligner.process_transcription_alignment(
        html_file, textgrid_file, audio_file, output_dir
    )

    if results['success']:
        print(f"   Archivos generados: {len(results['output_files'])}")
        for output_file in results['output_files']:
            print(f"   - {output_file}")
    else:
        print("Error, Algo salio mal ...")
        print(f"   Error: {results.get('error', 'Error desconocido')}")

    return results['success']

def test_all_html_files():
    """Probar con todos los HTML del entrenamiento"""
    training_dir = "data/training/202003301830"
    textgrid_file = os.path.join(training_dir, "202003301830.TextGrid")
    audio_file = os.path.join(training_dir, "202003301830.mp3")
    html_files = []
    for i in range(1, 12):  # 01.html to 11.html
        html_file = os.path.join(training_dir, f"{i:02d}.html")
        if os.path.exists(html_file):
            html_files.append(html_file)

    print(f"=== PROBANDO TODOS LOS HTML ({len(html_files)} archivos) ===")
    aligner = TranscriptionAligner()
    all_results = []
    for i, html_file in enumerate(html_files, 1):
        print(f"\nArchivo {i}/{len(html_files)}: {os.path.basename(html_file)}")
        html_text = aligner.parse_transcription_html(html_file)
        result = {
            'file': os.path.basename(html_file),
            'text_length': len(html_text),
            'preview': html_text[:100] if html_text else "No se extrajo texto"
        }
        all_results.append(result)
        print(f"   Longitud del texto: {result['text_length']} caracteres")
        print(f"   Vista previa: {result['preview']}...")
    summary_output = "output/html_parsing_summary.txt"
    os.makedirs(os.path.dirname(summary_output), exist_ok=True)

    with open(summary_output, 'w', encoding='utf-8') as f:
        f.write("Resumen de archivos HTML\n")
        f.write("=" * 30 + "\n\n")
        for result in all_results:
            f.write(f"Archivo: {result['file']}\n")
            f.write(f"Longitud del Texto: {result['text_length']} caracteres\n")
            f.write(f"Vista Previa: {result['preview']}\n")
            f.write("-" * 50 + "\n\n")
    print(f"\nResumen guardado en: {summary_output}")
    return all_results

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "all":
        test_all_html_files()
    else:
        test_training_data()
