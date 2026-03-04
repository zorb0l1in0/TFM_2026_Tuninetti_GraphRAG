"""
schemas.py
----------
Modelos Pydantic para los dos pasos del pipeline NER.
Usados por JsonOutputParser de LangChain.
"""

from pydantic import BaseModel, Field


# ── Paso 1: Detección de spans ────────────────────────────────────────────────

class SpanItem(BaseModel):
    text: str  = Field(description="Texto exacto del span en el documento")
    start: int = Field(description="Índice de carácter de inicio")
    end: int   = Field(description="Índice de carácter de fin")

class Paso1Salida(BaseModel):
    spans: list[SpanItem]  = Field(description="Lista de spans candidatos")
    razonamiento: str      = Field(description="Breve explicación de las decisiones")


# ── Paso 2: Clasificación + Relaciones ───────────────────────────────────────

class EntidadItem(BaseModel):
    text: str        = Field(description="Texto del span")
    entity_type: str = Field(description="Tipo de entidad de la ontología o NONE")
    confidence: str  = Field(description="high | medium | low")
    descripcion: str = Field(description="Breve justificación de la clasificación")

class RelacionItem(BaseModel):
    sujeto:   str = Field(description="Texto del span sujeto")
    relacion: str = Field(description="Nombre de la relación en la ontología")
    objeto:   str = Field(description="Texto del span objeto")

class Paso2Salida(BaseModel):
    entidades:  list[EntidadItem]  = Field(description="Entidades clasificadas")
    relaciones: list[RelacionItem] = Field(description="Relaciones entre entidades")