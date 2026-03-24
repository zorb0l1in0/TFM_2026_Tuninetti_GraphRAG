"""
schemas.py
----------
Modelos Pydantic para los dos pasos del pipeline NER.
Usados por JsonOutputParser de LangChain.
"""
from typing import Optional

from pydantic import BaseModel, Field


# ── Paso 1: Detección de spans ────────────────────────────────────────────────

class SpanItem(BaseModel):
    text: str  = Field(description="Texto exacto del span en el documento")

class Paso1Salida(BaseModel):
    spans: list[SpanItem]  = Field(description="Lista de spans candidatos")


# ── Paso 2: Clasificación + Relaciones ───────────────────────────────────────

class EntidadItem(BaseModel):
    text:        str = Field(description="Texto exacto del span")
    entity_type: str = Field(description="Tipo de entidad de la ontología o NONE")
    confidence:  str = Field(description="high | medium | low")
    descripcion: str = Field(description="Descripción breve en español de qué es esta entidad en el contexto del texto. Obligatorio, nunca vacío.")

class RelacionItem(BaseModel):
    sujeto:   str = Field(description="Texto del span sujeto")
    relacion: str = Field(description="Nombre de la relación en la ontología")
    objeto:   str = Field(description="Texto del span objeto")

class Paso2Salida(BaseModel):
    entidades:  list[EntidadItem]  = Field(description="Entidades clasificadas")
    relaciones: list[RelacionItem] = Field(description="Relaciones entre entidades")