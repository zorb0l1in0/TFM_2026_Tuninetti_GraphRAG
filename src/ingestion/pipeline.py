"""
pipeline.py
-----------
IngestionPipeline: orchestra ChunkerSeccionesMarkdown + GeneradorEmbeddingsMarkdown.

Flusso:
    cartella .md  →  chunk_documento()  →  generar()  →  CSV

Uso rapido:
    from ingestion.pipeline import IngestionPipeline

    pipeline = IngestionPipeline(api_key="sk-...")
    df = pipeline.ejecutar(
        carpeta_entrada="data/raw",
        archivo_salida="data/processed/chunks.csv",
    )

Uso avanzato (solo chunking, senza API):
    df = pipeline.solo_chunking("data/raw")

Uso da riga di comando:
    python -m ingestion.pipeline -i data/raw -o data/processed/chunks.csv
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from .chunker import ChunkerSeccionesMarkdown
from .embedder import GeneradorEmbeddingsMarkdown


class IngestionPipeline:
    """
    Orchestra chunking → embeddings → CSV.

    Args:
        api_key:              Chiave OpenAI (richiesta solo per gli embeddings)
        modelo_embeddings:    Modello da usare (default: text-embedding-3-small)
        max_caracteres_chunk: Limite caratteri per chunk (default: 1500)
        batch_size:           Dimensione batch chiamate API (default: 20)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        modelo_embeddings: str = "text-embedding-3-small",
        max_caracteres_chunk: int = 1500,
        batch_size: int = 20,
    ):
        self.max_caracteres_chunk = max_caracteres_chunk
        self._api_key             = api_key
        self._modelo              = modelo_embeddings
        self._batch_size          = batch_size

        self.chunker  = ChunkerSeccionesMarkdown(max_caracteres=max_caracteres_chunk)
        self._embedder: Optional[GeneradorEmbeddingsMarkdown] = None

    # ── API pubblica ──────────────────────────────────────────────────────────

    def ejecutar(
        self,
        carpeta_entrada: str,
        archivo_salida: str,
        max_chunks_por_doc: Optional[int] = None,
        incluir_metadata: bool = True,
    ) -> pd.DataFrame:
        """
        Pipeline completo: Markdown → chunks → embeddings → CSV.

        Returns:
            DataFrame salvato in archivo_salida
        """
        chunks   = self._chunking(carpeta_entrada, max_chunks_por_doc)
        embedder = self._get_embedder()
        df       = embedder.generar(chunks, incluir_metadata=incluir_metadata)
        embedder.guardar_csv(df, archivo_salida)
        print(f"\n🎉 Pipeline completada — {len(df)} chunks en {archivo_salida}")
        return df

    def solo_chunking(
        self,
        carpeta_entrada: str,
        archivo_salida: Optional[str] = None,
        max_chunks_por_doc: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Solo chunking, senza embeddings. Non richiede API key.

        Returns:
            DataFrame con i chunks (senza colonna 'embedding')
        """
        chunks = self._chunking(carpeta_entrada, max_chunks_por_doc)
        df     = pd.DataFrame(chunks)
        print(f"✅ {len(df)} chunks generati (senza embeddings)")

        if archivo_salida:
            salida = Path(archivo_salida)
            salida.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(salida, index=False, encoding="utf-8")
            print(f"💾 CSV guardado: {salida}")

        return df

    # ── Chunking interno ──────────────────────────────────────────────────────

    def _chunking(
        self,
        carpeta_entrada: str,
        max_chunks_por_doc: Optional[int],
    ) -> List[Dict]:
        carpeta     = Path(carpeta_entrada)
        archivos_md = list(carpeta.glob("**/*.md"))

        if not archivos_md:
            raise FileNotFoundError(f"Nessun file .md in: {carpeta}")

        print(f"📂 {len(archivos_md)} archivos Markdown in {carpeta}")

        todos: List[Dict] = []
        for archivo in tqdm(archivos_md, desc="Chunking"):
            chunks = self._procesar_archivo(archivo)
            if max_chunks_por_doc:
                chunks = chunks[:max_chunks_por_doc]
            todos.extend(chunks)

        print(f"📄 {len(todos)} chunks totali")
        return todos

    def _procesar_archivo(self, archivo: Path) -> List[Dict]:
        try:
            contenido = archivo.read_text(encoding="utf-8")
        except Exception as e:
            print(f"⚠️ {archivo.name}: {e}")
            return []

        titulo = archivo.stem
        for linea in contenido.splitlines():
            if linea.startswith("# "):
                titulo = linea[2:].strip()
                break

        tipo          = archivo.parent.name if archivo.parent.name != "." else "Desconocido"
        datos_chunks  = self.chunker.chunk_documento(
            texto=contenido,
            metadata={"titulo": titulo, "tipo": tipo, "archivo_origen": archivo.name},
        )
        total = len(datos_chunks)

        return [
            {
                "texto":          cd["texto"],
                "titulo":         titulo,
                "tipo":           tipo,
                "pagina":         i,
                "id_chunk":       f"{archivo.stem}_{i:04d}",
                "archivo_origen": archivo.name,
                "ruta_completa":  str(archivo),
                "indice_chunk":   i,
                "total_chunks":   total,
                "titulo_seccion": cd.get("titulo_seccion", ""),
                "caracteres":     cd["caracteres"],
                "palabras":       cd["palabras"],
            }
            for i, cd in enumerate(datos_chunks)
        ]

    # ── Lazy embedder ─────────────────────────────────────────────────────────

    def _get_embedder(self) -> GeneradorEmbeddingsMarkdown:
        if not self._api_key:
            raise ValueError(
                "api_key richiesta. Passa api_key nel costruttore "
                "o usa solo_chunking() per lavorare senza API."
            )
        if self._embedder is None:
            self._embedder = GeneradorEmbeddingsMarkdown(
                api_key=self._api_key,
                modelo=self._modelo,
                batch_size=self._batch_size,
            )
        return self._embedder


# ── CLI ───────────────────────────────────────────────────────────────────────

_DEFAULT_INPUT  = "../../data/raw"
_DEFAULT_OUTPUT = "../../data/processed/chunks/chunks_con_embeddings.csv"
_DEFAULT_CHARS  = 1500


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ingestion pipeline: Markdown → chunks → embeddings → CSV"
    )
    p.add_argument("-i", "--input",  required=True, help="Carpeta con archivos .md")
    p.add_argument("-o", "--output", required=True, help="CSV de salida")
    p.add_argument("-k", "--api-key", default=None,
                   help="Clave OpenAI (o variable OPENAI_API_KEY)")
    p.add_argument("-m", "--modelo", default="text-embedding-3-small",
                   choices=["text-embedding-3-small", "text-embedding-3-large",
                            "text-embedding-ada-002"])
    p.add_argument("--batch-size",     type=int, default=20)
    p.add_argument("--max-caracteres", type=int, default=_DEFAULT_CHARS)
    p.add_argument("--max-chunks",     type=int, default=None,
                   help="Máximo chunks por documento")
    p.add_argument("--solo-chunking",  action="store_true",
                   help="Solo chunking, sin embeddings")
    return p


def _pycharm_mode():
    """Modalità interattiva quando lanciato da PyCharm senza argomenti."""
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")

    print("=" * 60)
    print("🚀 INGESTION PIPELINE")
    print("=" * 60)
    print(f"\n  📂 Input  : {_DEFAULT_INPUT}")
    print(f"  📄 Output : {_DEFAULT_OUTPUT}")
    print(f"  📏 Chars  : {_DEFAULT_CHARS}")
    print(f"  🔑 API Key: {os.getenv('OPENAI_API_KEY', 'No definida')[:10]}...")
    print()
    print("  [1] Pipeline completo (chunking + embeddings)")
    print("  [2] Solo chunking (sin API key)")
    print()

    modo = input("Selecciona (1/2): ").strip()
    sys.argv = [sys.argv[0],
                "--input",  _DEFAULT_INPUT,
                "--output", _DEFAULT_OUTPUT]

    if modo == "1":
        api = os.getenv("OPENAI_API_KEY", "")
        if api:
            sys.argv += ["--api-key", api]
    elif modo == "2":
        sys.argv += ["--solo-chunking"]
    else:
        print("❌ Modo no válido")
        sys.exit(1)


def main():
    if len(sys.argv) == 1:
        _pycharm_mode()

    args    = _build_parser().parse_args()
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")

    pipeline = IngestionPipeline(
        api_key=api_key,
        modelo_embeddings=args.modelo,
        max_caracteres_chunk=args.max_caracteres,
        batch_size=args.batch_size,
    )

    if args.solo_chunking:
        pipeline.solo_chunking(
            carpeta_entrada=args.input,
            archivo_salida=args.output,
            max_chunks_por_doc=args.max_chunks,
        )
    else:
        pipeline.ejecutar(
            carpeta_entrada=args.input,
            archivo_salida=args.output,
            max_chunks_por_doc=args.max_chunks,
        )


if __name__ == "__main__":
    main()