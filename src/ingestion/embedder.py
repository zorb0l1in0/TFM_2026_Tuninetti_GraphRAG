#!/usr/bin/env python3
"""
Script para generar embeddings desde documentos Markdown usando un cliente centralizado de embeddings
y guardarlos en formato CSV compatible con el CSVLoader proporcionado.

Uso:
    python generar_embeddings.py --input carpeta_markdown --output salida.csv --api-key "tu-clave"

Ejemplo:
    python generar_embeddings.py -i ./documentos -o ./data/processed/chunks/documentos_con_embeddings.csv -k "sk-..."
"""

import os
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Dict
import json
from tqdm import tqdm
import hashlib
from datetime import datetime
import time
from dotenv import load_dotenv
from src.ingestion import ChunkerSeccionesMarkdown
from src.common.clients import get_embeddings_client, EMBEDDING_MODEL


class GeneradorEmbeddingsMarkdown:
    """
    Generador de embeddings para documentos Markdown usando el cliente centralizado
    """

    def __init__(self,
                 modelo: str = EMBEDDING_MODEL,
                 batch_size: int = 20,
                 max_retries: int = 3,
                 delay_segundos: float = 0.5,
                 max_caracteres_chunk: int = 1500):  # 👈 NUEVO
        self.modelo = modelo
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.delay_segundos = delay_segundos

        # 👇 Inicializar chunker semántico
        self.chunker = ChunkerSeccionesMarkdown(
            max_caracteres=max_caracteres_chunk,
            niveles_heading=[1, 2, 3]
        )


        print(f"🚀 Inicializando generador de embeddings con modelo: {modelo}")
        print(f"📏 Chunker semántico con máximo de {max_caracteres_chunk} caracteres por chunk")
        self.client = get_embeddings_client()


        # Verificar que la API key funciona
        self._verificar_conexion()

    def _verificar_conexion(self):
        """Verifica que la API key es válida"""
        try:
            # Hacer una llamada mínima para verificar
            self.client.embeddings.create(
                model=self.modelo,
                input="test",
                encoding_format="float"
            )
            print("✅ Conexión con el cliente de embeddings establecida correctamente")
        except Exception as e:
            print(f"❌ Error al conectar con el cliente de embeddings: {e}")
            raise

    def procesar_carpeta(self,
                         carpeta_entrada: str,
                         archivo_salida: str,
                         incluir_metadata: bool = True,
                         max_chunks_por_doc: int = None):
        """
        Procesa todos los archivos Markdown en una carpeta y genera embeddings.

        Args:
            carpeta_entrada: Ruta a la carpeta con archivos .md
            archivo_salida: Ruta donde guardar el CSV
            incluir_metadata: Si incluir metadatos adicionales
            max_chunks_por_doc: Límite de chunks por documento (None = sin límite)
        """
        carpeta = Path(carpeta_entrada)
        if not carpeta.exists():
            raise FileNotFoundError(f"❌ Carpeta no encontrada: {carpeta}")

        # Encontrar todos los archivos .md
        archivos_md = list(carpeta.glob("**/*.md"))
        print(f"📂 Encontrados {len(archivos_md)} archivos Markdown")

        if not archivos_md:
            print("⚠️ No se encontraron archivos .md en la carpeta especificada")
            return

        # Lista para almacenar todos los chunks
        todos_chunks = []

        # Procesar cada archivo
        for archivo in tqdm(archivos_md, desc="Procesando documentos"):
            chunks = self._procesar_archivo(
                archivo,
                incluir_metadata
            )

            # Aplicar límite si existe
            if max_chunks_por_doc and len(chunks) > max_chunks_por_doc:
                print(f"⚠️ {archivo.name}: {len(chunks)} chunks, limitando a {max_chunks_por_doc}")
                chunks = chunks[:max_chunks_por_doc]

            todos_chunks.extend(chunks)

        print(f"📄 Total chunks generados: {len(todos_chunks)}")

        # Generar embeddings por lotes
        print("🔢 Generando embeddings con el cliente centralizado...")
        df = self._generar_embeddings_dataframe(todos_chunks)

        # Guardar CSV
        self._guardar_csv(df, archivo_salida)

        return df

    def _procesar_archivo(self,
                          archivo: Path,
                          incluir_metadata: bool) -> List[Dict]:
        """
        Procesa un archivo Markdown y lo divide en chunks USANDO EL CHUNKER SEMÁNTICO.
        """
        try:
            with open(archivo, 'r', encoding='utf-8') as f:
                contenido = f.read()
        except Exception as e:
            print(f"⚠️ Error leyendo {archivo.name}: {e}")
            return []

        # Extraer título del archivo o primer heading
        titulo = archivo.stem
        lineas = contenido.split('\n')
        for linea in lineas:
            if linea.startswith('# '):
                titulo = linea[2:].strip()
                break

        # Extraer tipo de documento (clase) de la carpeta
        tipo = archivo.parent.name if archivo.parent.name != "." else "Desconocido"

        # 👇 USAR CHUNKER SEMÁNTICO (NO _dividir_texto)
        datos_chunks = self.chunker.chunk_documento(
            texto=contenido,
            metadata={
                "titulo": titulo,
                "tipo": tipo,
                "archivo_origen": archivo.name,
                "ruta_completa": str(archivo)
            }
        )

        # Convertir al formato esperado
        chunks = []
        for i, chunk_data in enumerate(datos_chunks):
            chunk = {
                "texto": chunk_data["texto"],
                "titulo": titulo,
                "tipo": tipo,
                "pagina": i,  # Simulamos página con índice
                "id_chunk": f"{archivo.stem}_{i:04d}",
                "archivo_origen": archivo.name,
                "ruta_completa": str(archivo),
                "indice_chunk": i,
                "total_chunks": len(datos_chunks),
                "titulo_seccion": chunk_data.get("titulo_seccion", ""),
                "caracteres": chunk_data["caracteres"],
                "palabras": chunk_data["palabras"]
            }

            # Añadir metadata adicional si se solicita
            if incluir_metadata:
                chunk.update({
                    "fecha_procesamiento": datetime.now().isoformat(),
                    "hash_contenido": hashlib.md5(chunk["texto"].encode()).hexdigest()[:8]
                })

            chunks.append(chunk)

        return chunks

    def _generar_embeddings_dataframe(self, chunks: List[Dict]) -> pd.DataFrame:
        """
        Genera embeddings para todos los chunks usando el cliente de embeddings.
        """
        # Extraer solo los textos
        textos = [chunk["texto"] for chunk in chunks]

        # Procesar por lotes
        embeddings = []

        for i in tqdm(range(0, len(textos), self.batch_size), desc="Embeddings"):
            batch = textos[i:i + self.batch_size]
            batch_embeddings = self._obtener_embeddings_batch(batch)
            embeddings.extend(batch_embeddings)

            # Pequeña pausa para respetar rate limits
            if i + self.batch_size < len(textos):
                time.sleep(self.delay_segundos)

        # Crear dataframe
        df = pd.DataFrame(chunks)

        # Convertir embeddings a string JSON (formato compatible con CSVLoader)
        df['embedding'] = [json.dumps(emb) for emb in embeddings]

        # Reordenar columnas para que coincida con lo esperado por CSVLoader
        columnas_esperadas = [
            'texto', 'titulo', 'tipo', 'pagina', 'id_chunk', 'archivo_origen',
            'ruta_completa', 'indice_chunk', 'total_chunks', 'embedding'
        ]

        # Añadir otras columnas que puedan existir
        otras_cols = [col for col in df.columns if col not in columnas_esperadas]
        columnas_finales = [col for col in columnas_esperadas if col in df.columns] + otras_cols

        return df[columnas_finales]

    def _obtener_embeddings_batch(self, textos: List[str]) -> List[List[float]]:
        """
        Obtiene embeddings para un batch de textos con reintentos.
        """
        for intento in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.modelo,
                    input=textos,
                    encoding_format="float"
                )
                # Extraer embeddings en orden
                return [item.embedding for item in response.data]

            except Exception as e:
                if intento < self.max_retries - 1:
                    wait_time = 2 ** intento  # Exponential backoff
                    print(f"⚠️ Error en batch, reintentando en {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    print(f"❌ Error fatal en batch después de {self.max_retries} intentos: {e}")
                    # Devolver embeddings vacíos como fallback
                    return [[] for _ in textos]

    def _guardar_csv(self, df: pd.DataFrame, archivo_salida: str):
        """
        Guarda el dataframe como CSV, creando directorios si es necesario.
        """
        salida = Path(archivo_salida)
        salida.parent.mkdir(parents=True, exist_ok=True)

        # Guardar CSV
        df.to_csv(salida, index=False, encoding='utf-8')

        print(f"\n✅ CSV guardado en: {salida}")
        print(f"📊 Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas")

        # Mostrar estadísticas básicas
        print(f"\n📋 Estadísticas:")
        if 'archivo_origen' in df.columns:
            print(f"  • Documentos únicos: {df['archivo_origen'].nunique()}")
        if 'tipo' in df.columns:
            print(f"  • Tipos de documento: {df['tipo'].nunique()}")
        print(f"  • Embeddings generados: {df['embedding'].notna().sum()}/{len(df)}")

        # Estadísticas de chunks
        if 'caracteres' in df.columns:
            print(
                f"  • Caracteres por chunk: min={df['caracteres'].min()}, max={df['caracteres'].max()}, media={df['caracteres'].mean():.0f}")

        # Ejemplo de embedding
        if 'embedding' in df.columns and len(df) > 0:
            primer_emb = df.iloc[0]['embedding']
            if isinstance(primer_emb, str):
                try:
                    emb_list = json.loads(primer_emb)
                    print(f"  • Dimensión embedding: {len(emb_list)}")
                except:
                    pass


def configurar_parser():
    """Configura el parser de argumentos"""
    parser = argparse.ArgumentParser(
        description="Genera embeddings desde archivos Markdown usando el cliente centralizado de embeddings"
    )

    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Carpeta con archivos Markdown de entrada"
    )

    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Archivo CSV de salida"
    )

    parser.add_argument(
        "-m", "--modelo",
        default=EMBEDDING_MODEL,
        choices=[EMBEDDING_MODEL],
        help=f"Modelo de embeddings (default: {EMBEDDING_MODEL})"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Tamaño de batch para la API (default: 20)"
    )

    parser.add_argument(
        "--max-caracteres",
        type=int,
        default=1500,
        help="Máximo de caracteres por chunk (default: 2000, ~500 tokens)"
    )

    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Máximo número de chunks por documento (default: sin límite)"
    )

    return parser


def main():
    """Función principal"""
    parser = configurar_parser()
    args = parser.parse_args()

    # Crear generador
    try:
        generador = GeneradorEmbeddingsMarkdown(
            modelo=args.modelo,
            batch_size=args.batch_size,
            max_caracteres_chunk=args.max_caracteres
        )

        # Procesar documentos
        generador.procesar_carpeta(
            carpeta_entrada=args.input,
            archivo_salida=args.output,
            max_chunks_por_doc=args.max_chunks
        )

        print(f"\n🎉 ¡Proceso completado exitosamente!")

    except Exception as e:
        print(f"\n❌ Error durante la ejecución: {e}")
        raise


if __name__ == "__main__":
    """
    Punto de entrada principal para ejecución directa desde PyCharm.
    Configura los valores por defecto para pruebas rápidas.
    """
    # Mostra dove cerca il .env
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"

    load_dotenv(env_path)

    print("=" * 60)
    print("🚀 GENERADOR DE EMBEDDINGS PARA DOCUMENTOS MARKDOWN")
    print("=" * 60)
    # Configuración para ejecución directa desde PyCharm
    _ROOT = Path(__file__).resolve().parent.parent.parent
    RUTA_ENTRADA_POR_DEFECTO = str(_ROOT / "data" / "raw")
    RUTA_SALIDA_POR_DEFECTO = str(_ROOT / "data" / "processed" / "chunks" / "chunks_con_embeddings.csv")

    # Verificar si se está ejecutando desde PyCharm sin argumentos
    import sys

    if len(sys.argv) == 1:
        print("\n📌 EJECUCIÓN DIRECTA DESDE PYCHARM")
        print("Usando configuración por defecto:\n")
        print(f"  📂 Entrada: {RUTA_ENTRADA_POR_DEFECTO}")
        print(f"  📄 Salida:  {RUTA_SALIDA_POR_DEFECTO}")
        print(f"  📏 Max caracteres: 2000 (~500 tokens)")
        print()

        # Preguntar si quiere continuar
        respuesta = input("¿Continuar con estos valores? (s/n): ").lower()
        if respuesta != 's':
            print("❌ Ejecución cancelada")
            sys.exit(0)

        # Configurar argumentos simulados
        sys.argv = [
            sys.argv[0],
            "--input", RUTA_ENTRADA_POR_DEFECTO,
            "--output", RUTA_SALIDA_POR_DEFECTO,
        ]

    # Ejecutar función principal
    main()
