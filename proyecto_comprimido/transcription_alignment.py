#!/usr/bin/env python3
# Sistema de Alineación de Transcripciones
import re
import os
import sys
import difflib
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

try:
    from sml import sml
except ImportError:
    sml = None


@dataclass
class SegmentoTranscripcion:
    tiempo_inicio: float
    tiempo_fin: float
    texto: str
    confianza: float = 1.0


@dataclass
class SegmentoError:
    tiempo_inicio: float
    tiempo_fin: float
    tipo_error: str
    texto_esperado: str
    texto_encontrado: str
    confianza: float


class AlineadorTranscripcion:

    def __init__(self):
        self.sml = sml

    def extraer_texto_html(self, directorio_html: str) -> str:
        """Extrae y combina texto de archivos HTML"""
        if not os.path.exists(directorio_html):
            return ""

        archivos_html = [f for f in os.listdir(directorio_html)
                        if f.endswith('.html') and f[:-5].isdigit()]
        archivos_html.sort(key=lambda x: int(x[:-5]))

        texto_combinado = []
        for archivo in archivos_html:
            ruta = os.path.join(directorio_html, archivo)
            with open(ruta, 'r', encoding='utf-8', errors='replace') as f:
                contenido = f.read()

            # Buscar contenido en div p-article__body
            patron = r'<div class="p-article__body">(.*?)</div>'
            match = re.search(patron, contenido, re.DOTALL)

            if match:
                texto = match.group(1)
                # Limpiar HTML
                texto = re.sub(r'<[^>]+>', '', texto)
                texto = re.sub(r'&[a-zA-Z0-9]+;', ' ', texto)
                texto = re.sub(r'\s+', ' ', texto)
                texto_combinado.append(texto.strip())

        return ' '.join(texto_combinado)

    def parsear_textgrid_ipu(self, ruta_textgrid: str) -> List[SegmentoTranscripcion]:
        """Extrae segmentos IPU del TextGrid"""
        try:
            with open(ruta_textgrid, 'r', encoding='utf-8', errors='replace') as f:
                contenido = f.read()
        except:
            return []

        lineas = contenido.split('\n')
        segmentos = []

        # Buscar tier IPU
        indice_ipu = -1
        for i, linea in enumerate(lineas):
            if '"IPU"' in linea.strip():
                indice_ipu = i + 1
                break

        if indice_ipu == -1:
            return []

        # Buscar número de intervalos
        i = indice_ipu
        while i < len(lineas):
            linea = lineas[i].strip()
            if linea.replace('.', '').isdigit() and float(linea) > 10:
                break
            i += 1

        if i >= len(lineas):
            return []

        try:
            num_intervalos = int(float(lineas[i].strip()))
            i += 1
        except:
            return []

        # Procesar intervalos
        procesados = 0
        while i < len(lineas) and procesados < num_intervalos:
            try:
                tiempo_inicio = float(lineas[i].strip())
                tiempo_fin = float(lineas[i + 1].strip())
                texto = lineas[i + 2].strip().strip('"')
                i += 3

                # Solo agregar segmentos con texto válido
                if texto and texto not in ['#', '_', '']:
                    segmento = SegmentoTranscripcion(
                        tiempo_inicio=tiempo_inicio,
                        tiempo_fin=tiempo_fin,
                        texto=texto.strip()
                    )
                    segmentos.append(segmento)

                procesados += 1
            except:
                i += 1
                continue

        return segmentos

    def normalizar_texto(self, texto: str) -> str:
        """Normaliza texto para comparación"""
        if not texto:
            return ""
        texto = texto.lower()
        texto = re.sub(r'[^\w\s]', '', texto)
        texto = re.sub(r'\s+', ' ', texto)
        return texto.strip()

    def calcular_similitud(self, texto1: str, texto2: str) -> float:
        """Calcula similitud entre textos"""
        if self.sml and texto1 and texto2:
            # Usar SML si está disponible
            palabras1 = texto1.split()
            palabras2 = texto2.split()
            if palabras1 and palabras2:
                vec1 = [float(len(palabra)) for palabra in palabras1[:20]]
                vec2 = [float(len(palabra)) for palabra in palabras2[:20]]

                # Igualar longitudes
                max_len = max(len(vec1), len(vec2))
                vec1.extend([0.0] * (max_len - len(vec1)))
                vec2.extend([0.0] * (max_len - len(vec2)))

                try:
                    error = float(self.sml.mae_metric(vec1, vec2))
                    max_val = max(max(vec1), max(vec2)) if vec1 and vec2 else 1.0
                    if max_val > 0:
                        similitud = 1.0 - min(1.0, error / max_val)
                    else:
                        similitud = 1.0 if error == 0 else 0.0
                    return max(0.0, min(1.0, similitud))
                except:
                    pass

        return difflib.SequenceMatcher(None, texto1, texto2).ratio()

    def crear_tier_transcripcion(self, texto_html: str, segmentos_ipu: List[SegmentoTranscripcion]) -> List[SegmentoTranscripcion]:
        """Crea tier Transc alineando texto HTML con timestamps IPU"""
        if not texto_html or not segmentos_ipu:
            return []

        palabras_html = self.normalizar_texto(texto_html).split()
        segmentos_transc = []
        indice_html = 0
        total_palabras_html = len(palabras_html)

        for seg_ipu in segmentos_ipu:
            if indice_html >= total_palabras_html:
                break

            # Estimar palabras HTML para este segmento IPU
            palabras_ipu = self.normalizar_texto(seg_ipu.texto).split()
            total_palabras_ipu = sum(len(self.normalizar_texto(s.texto).split()) for s in segmentos_ipu)

            if total_palabras_ipu > 0:
                ratio_palabras = len(palabras_ipu) / total_palabras_ipu
                palabras_estimadas = max(1, int(total_palabras_html * ratio_palabras))
            else:
                palabras_estimadas = 1

            # Extraer palabras HTML correspondientes
            fin_indice = min(indice_html + palabras_estimadas, total_palabras_html)
            palabras_segmento = palabras_html[indice_html:fin_indice]

            if palabras_segmento:
                texto_transc = ' '.join(palabras_segmento)
                confianza = self.calcular_similitud(texto_transc, seg_ipu.texto)

                segmento_transc = SegmentoTranscripcion(
                    tiempo_inicio=seg_ipu.tiempo_inicio,
                    tiempo_fin=seg_ipu.tiempo_fin,
                    texto=texto_transc,
                    confianza=confianza
                )
                segmentos_transc.append(segmento_transc)
                indice_html = fin_indice

        return segmentos_transc

    def detectar_errores(self, segmentos_transc: List[SegmentoTranscripcion],
                        segmentos_ipu: List[SegmentoTranscripcion]) -> List[SegmentoError]:
        """Detecta errores entre transcripción e IPU"""
        errores = []
        min_segmentos = min(len(segmentos_transc), len(segmentos_ipu))

        for i in range(min_segmentos):
            seg_transc = segmentos_transc[i]
            seg_ipu = segmentos_ipu[i]

            # Normalizar textos
            texto_transc_norm = self.normalizar_texto(seg_transc.texto)
            texto_ipu_norm = self.normalizar_texto(seg_ipu.texto)

            # Calcular similitud
            similitud = self.calcular_similitud(texto_transc_norm, texto_ipu_norm)

            # Detectar errores si similitud es baja
            if similitud < 0.7:
                tipo_error = self.clasificar_error(texto_transc_norm, texto_ipu_norm)

                error = SegmentoError(
                    tiempo_inicio=seg_transc.tiempo_inicio,
                    tiempo_fin=seg_transc.tiempo_fin,
                    tipo_error=tipo_error,
                    texto_esperado=seg_transc.texto,
                    texto_encontrado=seg_ipu.texto,
                    confianza=1.0 - similitud
                )
                errores.append(error)

        return errores

    def clasificar_error(self, texto_transc: str, texto_ipu: str) -> str:
        """Clasifica el tipo de error según requerimientos específicos"""
        palabras_transc = texto_transc.split()
        palabras_ipu = texto_ipu.split()

        # Calcular diferencias
        set_transc = set(palabras_transc)
        set_ipu = set(palabras_ipu)

        palabras_presentes_audio_no_transcripcion = set_ipu - set_transc
        palabras_presentes_transcripcion_no_audio = set_transc - set_ipu
        palabras_comunes = set_transc & set_ipu

        # 1. Palabras presentes en audio pero no en transcripción (palabras faltantes en transcripción)
        if len(palabras_presentes_audio_no_transcripcion) > len(palabras_presentes_transcripcion_no_audio):
            muestra = list(palabras_presentes_audio_no_transcripcion)[:3]
            return f"palabras_faltantes_en_transcripcion: {', '.join(muestra)}"

        # 2. Palabras presentes en transcripción pero no en audio
        elif len(palabras_presentes_transcripcion_no_audio) > len(palabras_presentes_audio_no_transcripcion):
            muestra = list(palabras_presentes_transcripcion_no_audio)[:3]
            return f"palabras_extra_en_transcripcion: {', '.join(muestra)}"

        # 3. Detectar palabras intercambiadas de posición
        elif len(palabras_transc) == len(palabras_ipu) and len(palabras_comunes) == len(set_transc):
            if palabras_transc != palabras_ipu:
                intercambios = []
                for i, (p_transc, p_ipu) in enumerate(zip(palabras_transc, palabras_ipu)):
                    if p_transc != p_ipu:
                        intercambios.append(f"pos{i}:{p_transc}→{p_ipu}")
                if intercambios:
                    return f"palabras_intercambiadas_posicion: {', '.join(intercambios[:2])}"

        # 4. Detectar palabras intercambiadas con otras (sustituciones)
        elif len(palabras_transc) == len(palabras_ipu) and len(palabras_comunes) < len(set_transc):
            sustituciones = []
            for i, (p_transc, p_ipu) in enumerate(zip(palabras_transc, palabras_ipu)):
                if p_transc != p_ipu:
                    sustituciones.append(f"{p_transc}→{p_ipu}")
            if sustituciones:
                return f"palabras_intercambiadas: {', '.join(sustituciones[:2])}"

        # 5. Discrepancia general
        else:
            return f"discrepancia_general: t{len(palabras_transc)}_a{len(palabras_ipu)}_comunes{len(palabras_comunes)}"

    def generar_textgrid(self, segmentos_transc: List[SegmentoTranscripcion],
                        segmentos_errores: List[SegmentoError],
                        ruta_salida: str) -> bool:
        """Genera TextGrid con tiers Transc y Errors"""
        try:
            tiempo_max = max([seg.tiempo_fin for seg in segmentos_transc]) if segmentos_transc else 1.0

            contenido = [
                'File type = "ooTextFile"',
                'Object class = "TextGrid"',
                '',
                '0',
                f'{tiempo_max}',
                '<exists>',
                '2'
            ]

            # Tier Transc
            contenido.extend([
                '"IntervalTier"',
                '"Transc"',
                '0',
                f'{tiempo_max}',
                f'{len(segmentos_transc)}'
            ])

            for segmento in segmentos_transc:
                contenido.extend([
                    f'{segmento.tiempo_inicio}',
                    f'{segmento.tiempo_fin}',
                    f'"{segmento.texto}"'
                ])

            # Tier Errors
            contenido.extend([
                '"IntervalTier"',
                '"Errors"',
                '0',
                f'{tiempo_max}',
                f'{len(segmentos_errores)}'
            ])

            for error in segmentos_errores:
                texto_error = f"{error.tipo_error}: Esperado='{error.texto_esperado}' | Encontrado='{error.texto_encontrado}'"
                contenido.extend([
                    f'{error.tiempo_inicio}',
                    f'{error.tiempo_fin}',
                    f'"{texto_error}"'
                ])

            with open(ruta_salida, 'w', encoding='utf-8') as f:
                f.write('\n'.join(contenido))

            return True
        except:
            return False

    def procesar_alineacion(self, dir_html: str, archivo_textgrid: str, dir_salida: str) -> Dict:
        """Procesa alineación completa"""
        try:
            os.makedirs(dir_salida, exist_ok=True)

            # Extraer texto HTML
            texto_html = self.extraer_texto_html(dir_html)
            if not texto_html:
                raise ValueError("No se pudo extraer texto HTML")

            # Parsear segmentos IPU
            segmentos_ipu = self.parsear_textgrid_ipu(archivo_textgrid)
            if not segmentos_ipu:
                raise ValueError("No se pudieron extraer segmentos IPU")

            # Crear tier transcripción
            segmentos_transc = self.crear_tier_transcripcion(texto_html, segmentos_ipu)

            # Detectar errores
            segmentos_errores = self.detectar_errores(segmentos_transc, segmentos_ipu)

            # Generar TextGrid
            ruta_textgrid_salida = os.path.join(dir_salida, "transcripcion_alineada.TextGrid")
            self.generar_textgrid(segmentos_transc, segmentos_errores, ruta_textgrid_salida)

            # Calcular métricas
            if segmentos_transc and segmentos_ipu:
                similitudes = []
                for i in range(min(len(segmentos_transc), len(segmentos_ipu))):
                    sim = self.calcular_similitud(
                        self.normalizar_texto(segmentos_transc[i].texto),
                        self.normalizar_texto(segmentos_ipu[i].texto)
                    )
                    similitudes.append(sim)

                similitud_promedio = sum(similitudes) / len(similitudes)
                tasa_errores = len(segmentos_errores) / len(segmentos_transc)
            else:
                similitud_promedio = 0.0
                tasa_errores = 0.0

            return {
                'segmentos_transc': segmentos_transc,
                'segmentos_ipu': segmentos_ipu,
                'segmentos_errores': segmentos_errores,
                'similitud_promedio': similitud_promedio,
                'tasa_errores': tasa_errores
            }

        except Exception as e:
            raise ValueError(f"Error de procesamiento: {e}")


def main():
    if len(sys.argv) < 4:
        print("Uso: python transcription_alignment_simple.py <dir_html> <archivo_textgrid> <dir_salida>")
        return

    dir_html = sys.argv[1]
    archivo_textgrid = sys.argv[2]
    dir_salida = sys.argv[3]

    print("=== SISTEMA DE ALINEACIÓN DE TRANSCRIPCIONES ===")
    print(f"HTML: {dir_html}")
    print(f"TextGrid: {archivo_textgrid}")
    print(f"Salida: {dir_salida}")

    alineador = AlineadorTranscripcion()

    try:
        resultado = alineador.procesar_alineacion(dir_html, archivo_textgrid, dir_salida)

        print("\n=== RESULTADOS ===")
        print(f"Segmentos transcripción: {len(resultado['segmentos_transc'])}")
        print(f"Segmentos IPU: {len(resultado['segmentos_ipu'])}")
        print(f"Errores detectados: {len(resultado['segmentos_errores'])}")
        print(f"Similitud promedio: {resultado['similitud_promedio']:.3f}")
        print(f"Tasa de errores: {resultado['tasa_errores']:.3f}")

        print(f"\nArchivo generado: transcripcion_alineada.TextGrid")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
