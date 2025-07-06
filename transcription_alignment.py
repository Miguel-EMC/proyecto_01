#!/usr/bin/env python3
"""
Transcription Alignment System
"""
import re
import os
import sys
import difflib
import subprocess
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
try:
    import textgrid
    TEXTGRID_AVAILABLE = True
except ImportError:
    print("Warning: textgrid module not available. Using manual parsing only.")
    TEXTGRID_AVAILABLE = False

try:
    from sml import sml
    SML_AVAILABLE = True
except ImportError:
    print("Warning: sml module not available. Some features may be limited.")
    SML_AVAILABLE = False
    sml = None


@dataclass
class TranscriptionSegment:
    """Representar un segmento de transcripción con tiempo y texto"""
    start_time: float
    end_time: float
    text: str
    confidence: float = 1.0

@dataclass
class AlignmentResult:
    """Resultado de alineación entre transcripción y IPU"""
    transcription_segments: List[TranscriptionSegment]
    ipu_segments: List[TranscriptionSegment]
    aligned_pairs: List[Tuple[int, int]]
    errors: List[Dict[str, Any]]

class TranscriptionAligner:
    """Sistema principal de alineación de transcripciones"""
    def __init__(self):
        self.sml = sml

    def parse_transcription_html(self, html_path: str) -> str:
        """
        Extraer y limpira el texto de transcripción de archivos HTML
        Args:
            html_path: Ruta al archivo HTML

        Returns:
            Texto limpio extraído del HTML
        """
        try:
            with open(html_path, 'r', encoding='utf-8') as file:
                content = file.read()
            body_pattern = r'<div class="p-article__body">(.*?)</div>'
            body_match = re.search(body_pattern, content, re.DOTALL)
            if not body_match:
                paragraph_pattern = r'<p[^>]*>(.*?)</p>'
                paragraphs = re.findall(paragraph_pattern, content, re.DOTALL)
                text = ' '.join(paragraphs)
            else:
                text = body_match.group(1)

            # Limpiar etiquetas HTML
            text = re.sub(r'<[^>]+>', '', text)
            # Limpiar espacios y caracteres especiales
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'&nbsp;', ' ', text)
            text = re.sub(r'&[a-zA-Z0-9]+;', ' ', text)
            return text.strip()
        except Exception as e:
            print(f"Error parsing HTML file {html_path}: {e}")
            return ""

    def parse_multiple_html_files(self, html_dir: str) -> str:
        """
        Extraer y combinar texto de múltiples archivos HTML numerados
        Args:
            html_dir: Directorio que contiene archivos HTML numerados (01.html, 02.html, etc.)
        Returns:
            Texto combinado de todos los archivos HTML
        """
        combined_text = []

        try:
            import os
            html_files = []
            for filename in os.listdir(html_dir):
                if filename.endswith('.html') and filename[:-5].isdigit():
                    html_files.append(filename)

            # Ordenar archivos por número
            html_files.sort(key=lambda x: int(x[:-5]))
            print(f"Encontrados en la carpeta {len(html_files)} archivos HTML: {html_files}")

            for filename in html_files:
                html_path = os.path.join(html_dir, filename)
                text = self.parse_transcription_html(html_path)
                if text:
                    combined_text.append(text)
                    print(f"Extraído de {filename}: {len(text)} caracteres")

            return ' '.join(combined_text)

        except Exception as e:
            print(f"Error parsing multiple HTML files from {html_dir}: {e}")
            return ""

    def extract_ipu_segments(self, textgrid_path: str) -> List[TranscriptionSegment]:
        """
        Extraer segmentos IPU del archivo TextGrid

        Args:
            textgrid_path: Ruta al archivo TextGrid

        Returns:
            Lista de segmentos IPU
        """
        try:
            ipu_segments = self._parse_textgrid_manually(textgrid_path)
            if ipu_segments:
                return ipu_segments
            if TEXTGRID_AVAILABLE:
                tg = textgrid.TextGrid.fromFile(textgrid_path)
                ipu_segments = []
                ipu_tier = None
                for tier in tg.tiers:
                    if tier.name.lower() == 'ipu':
                        ipu_tier = tier
                        break
                if not ipu_tier:
                    print(f"No se encontró tier IPU en {textgrid_path}")
                    return []
                for interval in ipu_tier:
                    if interval.mark and interval.mark.strip() != '#':
                        segment = TranscriptionSegment(
                            start_time=interval.minTime,
                            end_time=interval.maxTime,
                            text=interval.mark.strip()
                        )
                        ipu_segments.append(segment)

                return ipu_segments
            else:
                print("textgrid library not available and manual parsing failed")
                return []
        except Exception as e:
            print(f"Error extracting IPU segments from {textgrid_path}: {e}")
            return []

    def _parse_textgrid_manually(self, textgrid_path: str) -> List[TranscriptionSegment]:
        """
        Parsear manualmente el archivo TextGrid para manejar problemas de codificación
        donde cada carácter está separado por espacios.

        Args:
            textgrid_path: Ruta al archivo TextGrid

        Returns:
            Lista de segmentos IPU extraídos manualmente
        """
        try:
            # Probar diferentes codificaciones, incluyendo UTF-16
            encodings = ['utf-16', 'utf-16-le', 'utf-16-be', 'utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            content = None
            encoding_used = None
            for encoding in encodings:
                try:
                    with open(textgrid_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    encoding_used = encoding
                    break
                except UnicodeDecodeError:
                    continue

            if content is None:
                print(f"No se pudo leer el archivo con ninguna codificación: {textgrid_path}")
                return []
            lines = content.split('\n')
            ipu_segments = []
            ipu_tier_found = False
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if line == '"IPU"':
                    ipu_tier_found = True
                    i += 1
                    break
                i += 1

            if not ipu_tier_found:
                print(f"No se encontró tier IPU en {textgrid_path}")
                print("Líneas alrededor de la ubicación esperada del tier IPU:")
                for j in range(max(0, 50), min(len(lines), 70)):
                    print(f"  Línea {j}: {repr(lines[j])}")
                return []
            while i < len(lines) and lines[i].strip() != '0':
                i += 1
            if i < len(lines):
                i += 1
            if i < len(lines):
                i += 1
            if i < len(lines):
                try:
                    count = int(lines[i].strip())
                    i += 1
                except ValueError:
                    count = 0
            else:
                count = 0
            interval_count = 0
            while i < len(lines) and interval_count < count:
                try:
                    start_time = float(lines[i].strip())
                    i += 1
                    end_time = float(lines[i].strip())
                    i += 1
                    text_line = lines[i].strip()
                    if text_line.startswith('"') and text_line.endswith('"'):
                        text = text_line[1:-1]
                    else:
                        text = text_line
                    i += 1
                    # Solo agregar segmentos no vacíos que no sean solo '#' o '_'
                    if text and text.strip() not in ['#', '_', '']:
                        segment = TranscriptionSegment(
                            start_time=start_time,
                            end_time=end_time,
                            text=text.strip()
                        )
                        ipu_segments.append(segment)
                    interval_count += 1

                except (ValueError, IndexError) as e:
                    print(f"Error parsing interval at line {i}: {e}")
                    i += 1
                    continue

            return ipu_segments

        except Exception as e:
            print(f"Error en parseo manual del TextGrid: {e}")
            return []

    def align_transcription_to_ipu(self, html_text: str, ipu_segments: List[TranscriptionSegment]) -> AlignmentResult:
        """
        Alinear el texto de transcripción con los segmentos IPU usando alineación global
        Args:
            html_text: Texto extraído del HTML
            ipu_segments: Lista de segmentos IPU
        Returns:
            Resultado de alineación
        """
        ipu_text = ' '.join([seg.text for seg in ipu_segments])

        # Normalizar textos para comparación
        html_normalized = self._normalize_text(html_text)
        ipu_normalized = self._normalize_text(ipu_text)
        print(f"HTML text length: {len(html_normalized)} characters")
        print(f"IPU text length: {len(ipu_normalized)} characters")

        if self.sml:
            # Convertir textos a vectores numéricos para comparación
            html_vec = [float(ord(c)) for c in html_normalized[:100]]  # Limitar longitud
            ipu_vec = [float(ord(c)) for c in ipu_normalized[:100]]

            # Igualar longitudes
            max_len = max(len(html_vec), len(ipu_vec))
            html_vec.extend([0.0] * (max_len - len(html_vec)))
            ipu_vec.extend([0.0] * (max_len - len(ipu_vec)))

            # Calcular similitud usando métricas de sml.py
            mae = float(self.sml.mae_metric(html_vec, ipu_vec))
            max_val = max(max(html_vec), max(ipu_vec)) if html_vec and ipu_vec else 1.0
            similarity_ratio = 1.0 - (mae / max_val) if max_val > 0 else 0.0
            similarity_ratio = max(0.0, min(1.0, similarity_ratio))
        else:
            # Fallback a difflib
            matcher = difflib.SequenceMatcher(None, html_normalized, ipu_normalized)
            similarity_ratio = matcher.ratio()

        print(f"Similarity ratio: {similarity_ratio:.3f}")

        aligned_pairs = []
        errors = []
        transcription_segments = []
        matching_blocks = matcher.get_matching_blocks()
        print(f"Found {len(matching_blocks)} matching blocks")

        # Crear segmentos de transcripción basados en IPU pero con texto HTML alineado
        html_words = html_normalized.split()
        ipu_words = ipu_normalized.split()

        if self.sml and len(html_words) > 0 and len(ipu_words) > 0:
            # Crear dataset con longitudes de palabras para análisis
            word_lengths = [[len(word), i] for i, word in enumerate(html_words)]

            if len(word_lengths) > 1:
                minmax = self.sml.dataset_minmax(word_lengths)
                normalized_lengths = self.sml.normalize_dataset(word_lengths.copy(), minmax)
                means = self.sml.column_means(normalized_lengths)
                print(f"Longitud promedio normalizada de palabras: {means[0]:.3f}")

        # Mapear palabras HTML a segmentos IPU usando posición relativa
        html_word_index = 0
        total_html_words = len(html_words)

        for segment in ipu_segments:
            if segment.text.strip() in ['#', '_', '']:
                continue
            segment_words = self._normalize_text(segment.text).split()
            # Calcular cuántas palabras HTML corresponden a este segmento
            if self.sml and len(segment_words) > 0 and len(ipu_words) > 0:
                segment_data = [[len(segment_words), len(ipu_words), total_html_words]]
                minmax = self.sml.dataset_minmax(segment_data)
                if minmax and len(minmax) == 2:
                    normalized_data = self.sml.normalize_dataset(segment_data.copy(), minmax)
                    segment_ratio = normalized_data[0][0] if normalized_data[0][0] > 0 else len(segment_words) / len(ipu_words)
                else:
                    segment_ratio = len(segment_words) / len(ipu_words)
            else:
                segment_ratio = len(segment_words) / len(ipu_words) if len(ipu_words) > 0 else 0
            expected_html_words = max(1, int(total_html_words * segment_ratio))

            # Extraer palabras HTML correspondientes
            end_index = min(html_word_index + expected_html_words, total_html_words)
            segment_html_words = html_words[html_word_index:end_index]

            if segment_html_words:
                # Crear segmento de transcripción con texto HTML alineado
                aligned_text = ' '.join(segment_html_words)
                trans_segment = TranscriptionSegment(
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    text=aligned_text
                )
                transcription_segments.append(trans_segment)
                if self.sml:
                    # Convertir textos a vectores para comparación
                    seg_text = ' '.join(segment_words)
                    seg_vec = [float(ord(c)) for c in seg_text[:50]]
                    align_vec = [float(ord(c)) for c in aligned_text[:50]]

                    # Igualar longitudes
                    max_len = max(len(seg_vec), len(align_vec))
                    seg_vec.extend([0.0] * (max_len - len(seg_vec)))
                    align_vec.extend([0.0] * (max_len - len(align_vec)))

                    # Calcular similitud
                    mae = float(self.sml.mae_metric(seg_vec, align_vec))
                    max_val = max(max(seg_vec), max(align_vec)) if seg_vec and align_vec else 1.0
                    segment_similarity = 1.0 - (mae / max_val) if max_val > 0 else 0.0
                    segment_similarity = max(0.0, min(1.0, segment_similarity))
                else:
                    segment_similarity = difflib.SequenceMatcher(
                        None,
                        ' '.join(segment_words),
                        aligned_text
                    ).ratio()

                if segment_similarity < 0.6:
                    error = {
                        'type': 'mismatch',
                        'html_words': segment_html_words,
                        'ipu_words': segment_words,
                        'similarity': segment_similarity,
                        'start_time': segment.start_time,
                        'end_time': segment.end_time
                    }
                    errors.append(error)

                html_word_index = end_index

        return AlignmentResult(
            transcription_segments=transcription_segments,
            ipu_segments=ipu_segments,
            aligned_pairs=aligned_pairs,
            errors=errors
        )

    def _normalize_text(self, text: str) -> str:
        """Normalizar texto para comparación """
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        if self.sml and len(text) > 0:
            char_values = [[float(ord(char))] for char in text]
            if len(char_values) > 1:
                minmax = self.sml.dataset_minmax(char_values)
                normalized_chars = self.sml.normalize_dataset(char_values, minmax)
                normalized_text = ''.join(chr(int(32 + val[0] * 94)) for val in normalized_chars)
                return normalized_text

        return text

    def generate_textgrid_transc(self, alignment_result: AlignmentResult, output_path: str) -> bool:
        """
        Generar un nuevo archivo TextGrid con tier Transc

        Args:
            alignment_result: Resultado de alineación
            output_path: Ruta de salida para el TextGrid

        Returns:
            True si se generó exitosamente
        """
        try:
            if not TEXTGRID_AVAILABLE:
                # Generar manualmente si la librería textgrid no está disponible
                return self._generate_textgrid_manually(alignment_result, output_path, "Transc")
            tg = textgrid.TextGrid()
            if not alignment_result.transcription_segments:
                max_time = 1.0
            else:
                max_time = max([seg.end_time for seg in alignment_result.transcription_segments])
            tg.maxTime = max_time
            # Crear tier Transc
            transc_tier = textgrid.IntervalTier(name="Transc", minTime=0, maxTime=max_time)
            for segment in alignment_result.transcription_segments:
                interval = textgrid.Interval(
                    minTime=segment.start_time,
                    maxTime=segment.end_time,
                    mark=segment.text
                )
                transc_tier.intervals.append(interval)
            tg.tiers.append(transc_tier)
            tg.write(output_path)
            return True
        except Exception as e:
            print(f"Error generando TextGrid: {e}")
            return False

    def detect_errors_with_espeak(self, transc_tier: List[TranscriptionSegment],
                                 ipu_tier: List[TranscriptionSegment]) -> List[Dict[str, Any]]:
        """
        Detectar errores usando espeak-ng para comparar pronunciación
        Args:
            transc_tier: Segmentos de transcripción
            ipu_tier: Segmentos IPU
        Returns:
            Lista de errores detectados
        """
        errors = []

        try:
            for i, (transc_seg, ipu_seg) in enumerate(zip(transc_tier, ipu_tier)):
                transc_phonemes = self._get_phonemes_espeak(transc_seg.text)
                ipu_phonemes = self._get_phonemes_espeak(ipu_seg.text)
                similarity = self._calculate_phoneme_similarity(transc_phonemes, ipu_phonemes)
                mae_error = 0.0
                rmse_error = 0.0
                combined_error = 0.0

                if self.sml and transc_phonemes and ipu_phonemes:
                    # Convertir a vectores numéricos
                    transc_vec = [float(ord(c)) for c in transc_phonemes[:20]]
                    ipu_vec = [float(ord(c)) for c in ipu_phonemes[:20]]
                    max_len = max(len(transc_vec), len(ipu_vec))
                    transc_vec.extend([0.0] * (max_len - len(transc_vec)))
                    ipu_vec.extend([0.0] * (max_len - len(ipu_vec)))
                    mae_error = float(self.sml.mae_metric(transc_vec, ipu_vec))
                    rmse_error = float(self.sml.rmse_metric(transc_vec, ipu_vec))
                    combined_error = (mae_error + rmse_error) / 2.0
                    error_threshold = 50.0  # Ajustar

                    should_flag_error = combined_error > error_threshold or similarity < 0.8
                else:
                    should_flag_error = similarity < 0.8

                if should_flag_error:
                    error = {
                        'segment_index': i,
                        'transc_text': transc_seg.text,
                        'ipu_text': ipu_seg.text,
                        'transc_phonemes': transc_phonemes,
                        'ipu_phonemes': ipu_phonemes,
                        'similarity': similarity,
                        'mae_error': mae_error,
                        'rmse_error': rmse_error,
                        'combined_error': combined_error,
                        'start_time': transc_seg.start_time,
                        'end_time': transc_seg.end_time
                    }
                    errors.append(error)

        except Exception as e:
            print(f"Error detecting errors with espeak: {e}")
        return errors

    def _get_phonemes_espeak(self, text: str) -> str:
        """Obtiener fonemas usando espeak-ng"""
        try:
            result = subprocess.run(
                ['espeak-ng', '-q', '--ipa', '-s', '100', text],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.stdout.strip()
        except:
            return text

    def _calculate_phoneme_similarity(self, phonemes1: str, phonemes2: str) -> float:
        """Calcular similitud entre fonemas usando sml.py"""
        if not phonemes1 or not phonemes2:
            return 0.0
        if self.sml:
            vec1 = [float(ord(c)) for c in phonemes1]
            vec2 = [float(ord(c)) for c in phonemes2]
            max_len = max(len(vec1), len(vec2))
            vec1.extend([0.0] * (max_len - len(vec1)))
            vec2.extend([0.0] * (max_len - len(vec2)))
            mae = float(self.sml.mae_metric(vec1, vec2))
            rmse = float(self.sml.rmse_metric(vec1, vec2))
            max_possible_error = max(max(vec1), max(vec2)) if vec1 and vec2 else 1.0
            similarity = 1.0 - (mae / max_possible_error) if max_possible_error > 0 else 0.0
            return max(0.0, min(1.0, similarity))

        # Fallback a difflib
        matcher = difflib.SequenceMatcher(None, phonemes1, phonemes2)
        return matcher.ratio()

    def generate_textgrid_transc_errors(self, errors: List[Dict[str, Any]],
                                       output_path: str) -> bool:
        """
        Generar tier TranscErrors con errores detectados
        Args:
            errors: Lista de errores
            output_path: Ruta de salida

        Returns:
            True si se generó exitosamente
        """
        try:
            if not TEXTGRID_AVAILABLE:
                return self._generate_errors_textgrid_manually(errors, output_path)
            tg = textgrid.TextGrid()
            if not errors:
                tg.maxTime = 1.0
                tg.write(output_path)
                return True
            max_time = max([error['end_time'] for error in errors])
            tg.maxTime = max_time
            errors_tier = textgrid.IntervalTier(name="TranscErrors", minTime=0, maxTime=max_time)
            for error in errors:
                error_text = f"Esperado: {error['transc_text']} | Encontrado: {error['ipu_text']} | Similarity: {error['similarity']:.2f}"
                interval = textgrid.Interval(
                    minTime=error['start_time'],
                    maxTime=error['end_time'],
                    mark=error_text
                )
                errors_tier.intervals.append(interval)
            tg.tiers.append(errors_tier)
            tg.write(output_path)
            return True
        except Exception as e:
            print(f"Error generando TranscErrors TextGrid: {e}")
            return False

    def play_and_validate(self, audio_path: str, error_segments: List[Dict[str, Any]]) -> str:
        """
        Facilitar validación manual de errores

        Args:
            audio_path: Ruta al archivo de audio
            error_segments: Lista de segmentos con errores
        """
        validation_report = []
        validation_report.append("=== REPORTE DE VALIDACIÓN MANUAL ===\n")
        validation_report.append(f"Archivo de audio: {audio_path}")
        validation_report.append(f"Total de errores detectados: {len(error_segments)}\n")
        for i, error in enumerate(error_segments, 1):
            validation_report.append(f"Error {i}:")
            start_time = error.get('start_time', 0.0)
            end_time = error.get('end_time', 1.0)
            validation_report.append(f"  Tiempo: {start_time:.2f}s - {end_time:.2f}s")
            # Manejar diferentes formatos de error
            if 'transc_text' in error and 'ipu_text' in error:
                # Formato espeak
                validation_report.append(f"  Transcripción esperada: '{error['transc_text']}'")
                validation_report.append(f"  Transcripción IPU: '{error['ipu_text']}'")
                validation_report.append(f"  Similitud fonética: {error.get('similarity', 0.0):.2f}")
            elif 'html_words' in error and 'ipu_words' in error:
                # Formato de alineación
                html_text = ' '.join(error['html_words']) if isinstance(error['html_words'], list) else str(error['html_words'])
                ipu_text = ' '.join(error['ipu_words']) if isinstance(error['ipu_words'], list) else str(error['ipu_words'])
                validation_report.append(f"  Texto HTML: '{html_text}'")
                validation_report.append(f"  Texto IPU: '{ipu_text}'")
                validation_report.append(f"  Similitud: {error.get('similarity', 0.0):.2f}")
            else:
                # Formato genérico
                validation_report.append(f"  Tipo de error: {error.get('type', 'unknown')}")
                validation_report.append(f"  Detalles: {str(error)}")
            validation_report.append(f"  Recomendación: Validar manualmente usando Praat")
            validation_report.append("")
        validation_report.append("=== INSTRUCCIONES DE VALIDACIÓN ===")
        validation_report.append("1. Abrir el archivo de audio en Praat")
        validation_report.append("2. Cargar el archivo TextGrid con errores")
        validation_report.append("3. Escuchar cada segmento marcado como error")
        validation_report.append("4. Confirmar o corregir las discrepancias detectadas")
        validation_report.append("5. Documentar hallazgos en este reporte")
        validation_report.append("")
        validation_report.append("=== RESUMEN ESTADÍSTICO ===")
        validation_report.append(f"- Errores de alineación: {len([e for e in error_segments if 'html_words' in e])}")
        validation_report.append(f"- Errores fonéticos: {len([e for e in error_segments if 'transc_text' in e])}")
        validation_report.append(f"- Similitud promedio: {sum([e.get('similarity', 0.0) for e in error_segments]) / len(error_segments):.3f}" if error_segments else "- Sin errores")
        return "\n".join(validation_report)

    def cross_validate_alignment(self, ipu_segments: List[TranscriptionSegment],
                                n_folds: int = 5) -> Dict[str, Any]:
        """
        Realizar validación cruzada del algoritmo de alineación usando sml.py
        Args:
            ipu_segments: Segmentos IPU
            n_folds: Número de pliegues para validación cruzada
        Returns:
            Diccionario con resultados de validación cruzada
        """
        if not self.sml or not ipu_segments:
            return {'error': 'sml.py no disponible o sin segmentos IPU'}

        try:
            # Preparar dataset para validación cruzada
            # Convertir segmentos a formato numérico
            dataset = []
            for seg in ipu_segments:
                if seg.text and seg.text.strip() not in ['#', '_', '']:
                    dataset.append([
                        seg.start_time,
                        seg.end_time,
                        len(seg.text),
                        float(ord(seg.text[0])) if seg.text else 0.0,  # Primer carácter como feature
                        len(seg.text.split())  # Número de palabras
                    ])

            if len(dataset) < n_folds:
                return {'error': f'Dataset muy pequeño para {n_folds} pliegues'}

            folds = self.sml.cross_validation_split(dataset, n_folds)
            fold_results = []
            for i, fold in enumerate(folds):
                test_fold = fold
                train_folds = [f for j, f in enumerate(folds) if j != i]
                train_data = [item for fold in train_folds for item in fold]

                # Evaluar usando métricas de sml.py
                if len(train_data) > 0 and len(test_fold) > 0:
                    # Extraer valores objetivo (duración)
                    test_targets = [row[1] - row[0] for row in test_fold]
                    # Usar algoritmo baseline de sml.py
                    predicted = self.sml.zero_rule_algorithm_regression(train_data, test_fold)
                    # Calcular métricas
                    mae = float(self.sml.mae_metric(test_targets, predicted))
                    rmse = float(self.sml.rmse_metric(test_targets, predicted))
                    fold_results.append({
                        'fold': i + 1,
                        'mae': mae,
                        'rmse': rmse,
                        'train_size': len(train_data),
                        'test_size': len(test_fold)
                    })

            # Calcular estadísticas agregadas
            if fold_results:
                avg_mae = sum(r['mae'] for r in fold_results) / len(fold_results)
                avg_rmse = sum(r['rmse'] for r in fold_results) / len(fold_results)

                return {
                    'success': True,
                    'n_folds': n_folds,
                    'avg_mae': avg_mae,
                    'avg_rmse': avg_rmse,
                    'fold_results': fold_results,
                    'dataset_size': len(dataset)
                }
            else:
                return {'error': 'No se pudieron calcular resultados de validación'}

        except Exception as e:
            return {'error': f'Error en validación cruzada: {str(e)}'}

    def evaluate_alignment_quality(self, alignment_result: AlignmentResult) -> Dict[str, Any]:
        """
        Evalúar la calidad de la alineación usando métricas de sml.py

        Args:
            alignment_result: Resultado de alineación

        Returns:
            Diccionario con métricas de calidad
        """
        if not self.sml or not alignment_result.transcription_segments:
            return {'error': 'sml.py no disponible o sin segmentos'}

        try:
            # Preparar datos para evaluación
            durations_transc = [seg.end_time - seg.start_time for seg in alignment_result.transcription_segments]
            durations_ipu = [seg.end_time - seg.start_time for seg in alignment_result.ipu_segments]

            # Igualar longitudes
            min_len = min(len(durations_transc), len(durations_ipu))
            durations_transc = durations_transc[:min_len]
            durations_ipu = durations_ipu[:min_len]

            if len(durations_transc) == 0:
                return {'error': 'No hay segmentos para evaluar'}

            # Calcular métricas usando sml.py
            mae = float(self.sml.mae_metric(durations_ipu, durations_transc))
            rmse = float(self.sml.rmse_metric(durations_ipu, durations_transc))

            # Calcular estadísticas adicionales
            dataset = [[d1, d2] for d1, d2 in zip(durations_transc, durations_ipu)]
            means = self.sml.column_means(dataset)
            stdevs = self.sml.column_stdevs(dataset, means)

            return {
                'success': True,
                'mae': mae,
                'rmse': rmse,
                'mean_duration_transc': means[0] if len(means) > 0 else 0.0,
                'mean_duration_ipu': means[1] if len(means) > 1 else 0.0,
                'stdev_duration_transc': stdevs[0] if len(stdevs) > 0 else 0.0,
                'stdev_duration_ipu': stdevs[1] if len(stdevs) > 1 else 0.0,
                'num_segments': len(durations_transc),
                'num_errors': len(alignment_result.errors)
            }

        except Exception as e:
            return {'error': f'Error evaluando calidad: {str(e)}'}

    def process_transcription_alignment(self, html_path: str, textgrid_path: str,
                                      audio_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Procesar alineación completa de transcripción

        Args:
            html_path: Ruta al archivo HTML o directorio con archivos HTML
            textgrid_path: Ruta al archivo TextGrid
            audio_path: Ruta al archivo de audio
            output_dir: Directorio de salida

        Returns:
            Diccionario con resultados del procesamiento
        """
        results = {
            'success': False,
            'html_text': '',
            'ipu_segments': [],
            'alignment_result': None,
            'errors': [],
            'output_files': [],
            'validation_report': ''
        }

        try:
            # Paso 1: Extraer texto del HTML
            print("Extrayendo texto de transcripción HTML...")
            if os.path.isdir(html_path):
                html_text = self.parse_multiple_html_files(html_path)
            else:
                html_text = self.parse_transcription_html(html_path)
            results['html_text'] = html_text
            if not html_text:
                print("No se pudo extraer texto del HTML")
                return results
            # Paso 2: Extraer segmentos IPU
            print("Extrayendo segmentos IPU...")
            ipu_segments = self.extract_ipu_segments(textgrid_path)
            results['ipu_segments'] = ipu_segments
            if not ipu_segments:
                print("No se pudieron extraer segmentos IPU")
                return results
            # Paso 3: Alinear transcripción con IPU
            print("Alineando transcripción con IPU...")
            alignment_result = self.align_transcription_to_ipu(html_text, ipu_segments)
            results['alignment_result'] = alignment_result
            # Paso 4: Generar TextGrid con tier Transc
            print("Generando TextGrid con tier Transc...")
            transc_output = os.path.join(output_dir, "transcription_aligned.TextGrid")
            if self.generate_textgrid_transc(alignment_result, transc_output):
                results['output_files'].append(transc_output)
                print(f"TextGrid Transc generado: {transc_output}")
            # Paso 5: Detectar errores con espeak y de alineación
            print("Detectando errores...")
            espeak_errors = self.detect_errors_with_espeak(
                alignment_result.transcription_segments,
                alignment_result.ipu_segments
            )
            # Combinar errores de alineación y espeak
            all_errors = alignment_result.errors.copy()
            for error in espeak_errors:
                if error not in all_errors:
                    all_errors.append(error)
            results['errors'] = all_errors
            # Paso 6: Generar TextGrid con errores
            print("Generando TextGrid con errores...")
            errors_output = os.path.join(output_dir, "transcription_errors.TextGrid")
            if self.generate_textgrid_transc_errors(all_errors, errors_output):
                results['output_files'].append(errors_output)
                print(f"TextGrid TranscErrors generado: {errors_output}")
            # Paso 7: Evaluar calidad de alineación
            print("Evaluando calidad de alineación...")
            quality_metrics = self.evaluate_alignment_quality(alignment_result)
            results['quality_metrics'] = quality_metrics
            # Paso 8: Validación cruzada si hay suficientes datos
            if len(ipu_segments) >= 5:
                print("Realizando validación cruzada...")
                cv_results = self.cross_validate_alignment(ipu_segments)
                results['cross_validation'] = cv_results
            # Paso 9: Generar reporte de validación
            print("Generando reporte de validación...")
            validation_report = self.play_and_validate(audio_path, all_errors)
            results['validation_report'] = validation_report
            # Guardar reporte
            report_output = os.path.join(output_dir, "validation_report.txt")
            with open(report_output, 'w', encoding='utf-8') as f:
                f.write(validation_report)
            results['output_files'].append(report_output)
            results['success'] = True
            print("Procesamiento completado exitosamente")
        except Exception as e:
            print(f"Error en procesamiento: {e}")
            results['error'] = str(e)

        return results

    # Opciónal si la liberia textgrid no esta disponible
    def _generate_textgrid_manually(self, alignment_result: AlignmentResult, output_path: str, tier_name: str) -> bool:
        """
        Generar TextGrid manualmente sin usar la librería textgrid
        """
        try:
            if not alignment_result.transcription_segments:
                max_time = 1.0
            else:
                max_time = max([seg.end_time for seg in alignment_result.transcription_segments])
            # Crear contenido de TextGrid manualmente
            content = []
            content.append('File type = "ooTextFile"')
            content.append('Object class = "TextGrid"')
            content.append('')
            content.append('0')
            content.append(f'{max_time}')
            content.append('<exists>')
            content.append('1')
            content.append('"IntervalTier"')
            content.append(f'"{tier_name}"')
            content.append('0')
            content.append(f'{max_time}')
            content.append(f'{len(alignment_result.transcription_segments)}')
            for segment in alignment_result.transcription_segments:
                content.append(f'{segment.start_time}')
                content.append(f'{segment.end_time}')
                content.append(f'"{segment.text}"')
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(content))
            return True
        except Exception as e:
            print(f"Error generando TextGrid manually: {e}")
            return False

    def _generate_errors_textgrid_manually(self, errors: List[Dict[str, Any]], output_path: str) -> bool:
        """
        Generar TextGrid de errores manualmente sin usar la librería textgrid
        """
        try:
            if not errors:
                max_time = 1.0
            else:
                max_time = max([error.get('end_time', 1.0) for error in errors])

            # Crear contenido de TextGrid manualmente
            content = []
            content.append('File type = "ooTextFile"')
            content.append('Object class = "TextGrid"')
            content.append('')
            content.append('0')
            content.append(f'{max_time}')
            content.append('<exists>')
            content.append('1')
            content.append('"IntervalTier"')
            content.append('"TranscErrors"')
            content.append('0')
            content.append(f'{max_time}')
            content.append(f'{len(errors)}')
            for error in errors:
                if 'transc_text' in error and 'ipu_text' in error:
                    # Formato espeak
                    error_text = f"Esperado: {error['transc_text']} | Encontrado: {error['ipu_text']} | Similarity: {error.get('similarity', 0.0):.2f}"
                    start_time = error.get('start_time', 0.0)
                    end_time = error.get('end_time', 1.0)
                elif 'html_words' in error and 'ipu_words' in error:
                    # Formato de alineación
                    html_text = ' '.join(error['html_words']) if isinstance(error['html_words'], list) else str(error['html_words'])
                    ipu_text = ' '.join(error['ipu_words']) if isinstance(error['ipu_words'], list) else str(error['ipu_words'])
                    error_text = f"HTML: {html_text} | IPU: {ipu_text} | Similarity: {error.get('similarity', 0.0):.2f}"
                    start_time = error.get('start_time', 0.0)
                    end_time = error.get('end_time', 1.0)
                else:
                    # Formato genérico
                    error_text = f"Error: {error.get('type', 'unknown')} | Details: {str(error)}"
                    start_time = error.get('start_time', 0.0)
                    end_time = error.get('end_time', 1.0)

                content.append(f'{start_time}')
                content.append(f'{end_time}')
                content.append(f'"{error_text}"')

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(content))

            return True

        except Exception as e:
            print(f"Error generando errors TextGrid manually: {e}")
            return False

def main():
    """Función principal"""
    if len(sys.argv) < 4:
        print("Uso: python transcription_alignment.py <html_file> <textgrid_file> <audio_file> [output_dir]")
        return

    html_file = sys.argv[1]
    textgrid_file = sys.argv[2]
    audio_file = sys.argv[3]
    output_dir = sys.argv[4] if len(sys.argv) > 4 else "output"
    os.makedirs(output_dir, exist_ok=True)
    aligner = TranscriptionAligner()
    results = aligner.process_transcription_alignment(
        html_file, textgrid_file, audio_file, output_dir
    )
    if results['success']:
        print(f"\n=== RESULTADOS ===")
        print(f"Segmentos IPU extraídos: {len(results['ipu_segments'])}")
        print(f"Errores detectados: {len(results['errors'])}")
        print(f"Archivos generados: {len(results['output_files'])}")
        for output_file in results['output_files']:
            print(f"  - {output_file}")
        if 'quality_metrics' in results and results['quality_metrics'].get('success'):
            qm = results['quality_metrics']
            print(f"\n=== MÉTRICAS DE CALIDAD (usando sml.py) ===")
            print(f"MAE (Error Absoluto Medio): {qm['mae']:.3f}")
            print(f"RMSE (Raíz del Error Cuadrático Medio): {qm['rmse']:.3f}")
            print(f"Duración promedio Transc: {qm['mean_duration_transc']:.3f}s")
            print(f"Duración promedio IPU: {qm['mean_duration_ipu']:.3f}s")
            print(f"Número de segmentos evaluados: {qm['num_segments']}")
        if 'cross_validation' in results and results['cross_validation'].get('success'):
            cv = results['cross_validation']
            print(f"\n=== VALIDACIÓN CRUZADA (usando sml.py) ===")
            print(f"Número de pliegues: {cv['n_folds']}")
            print(f"MAE promedio: {cv['avg_mae']:.3f}")
            print(f"RMSE promedio: {cv['avg_rmse']:.3f}")
            print(f"Tamaño del dataset: {cv['dataset_size']}")
        print(f"\n=== REPORTE DE VALIDACIÓN ===")
        print(results['validation_report'])
    else:
        print(f"Error en procesamiento: {results.get('error', 'Error desconocido')}")

if __name__ == "__main__":
    main()
