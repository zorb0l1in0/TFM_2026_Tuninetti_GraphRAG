"""
ingestion/
----------
Pipeline di ingestion: Markdown → Chunks → Embeddings → CSV.

Uso rapido:
    from ingestion import IngestionPipeline

    df = IngestionPipeline(api_key="sk-...").ejecutar(
        carpeta_entrada="data/raw",
        archivo_salida="data/processed/chunks.csv",
    )
"""

from .chunker  import ChunkerSeccionesMarkdown
from .embedder import GeneradorEmbeddingsMarkdown
from .pipeline import IngestionPipeline

__all__ = [
    "ChunkerSeccionesMarkdown",
    "GeneradorEmbeddingsMarkdown",
    "IngestionPipeline",
]