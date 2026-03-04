"""
ner/
----
Pipeline NER de dos pasos guiada por ontología.
Stack: LangChain + OpenAI GPT-4o
"""

from .schemas import SpanItem, Paso1Salida, EntidadItem, RelacionItem, Paso2Salida
from .ontologia import CargadorOntologia
from .verificador import VerificadorRestricciones
from .cargador_chunks import CargadorChunksCSV
from .pipeline import PipelineNERDosPasos

__all__ = [
    "SpanItem",
    "Paso1Salida",
    "EntidadItem",
    "RelacionItem",
    "Paso2Salida",
    "CargadorOntologia",
    "VerificadorRestricciones",
    "CargadorChunksCSV",
    "PipelineNERDosPasos",
]