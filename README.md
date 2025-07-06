# Proyecto de Alineación de Transcripciones
Estudiante: Miguel Muzo
Materia: DATA MINING Y MACHINE LEARNING (ICCD623)

Este proyecto en Python alinea transcripciones de audio con segmentos de habla usando ASR y VAD.

## Como ejecutar

### Opción 1: Script principal
```bash
python ejecutar.py
```

### Opción 2: Manual  
```bash
pip install -r requirements.txt

python transcription_alignment.py \
    data/training/202003301830/01.html \
    data/training/202003301830/202003301830.TextGrid \
    data/training/202003301830/202003301830.mp3 \
    output
```

## Archivos que se generan

En la carpeta `output/` se crean:

1. `transcription_aligned.TextGrid` - Transcripción alineada
2. `transcription_errors.TextGrid` - Errores encontrados  
3. `validation_report.txt` - Reporte de validación

## Que hace

- Alinea texto HTML con IPUs usando difflib
- Detecta errores con espeak-ng
- Genera archivos TextGrid para Praat
- Crea reportes para validación manual

## Resultados

- 184 segmentos IPU extraídos
- 2,050 segmentos de transcripción alineados
- 183 errores detectados
- Tiempo: 3-5 segundos

## Archivos principales

- transcription_alignment.py - Script principal
- sml.py - Librería SML
- test_system.py - Pruebas
- ejecutar.py - Script automático
- requirements.txt - Dependencias

## Dependencias

- Python 3.9+
- textgrid==1.5
- numpy
- espeak-ng

## Validación con Praat

1. Descargar Praat
2. Abrir audio y TextGrid de errores
3. Escuchar segmentos marcados
4. Validar con el reporte

## Problemas comunes

- Error "espeak-ng not found": instalar con `sudo apt-get install espeak-ng`
- Error "textgrid not found": ejecutar `pip install -r requirements.txt`
