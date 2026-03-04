"""
prompts.py
----------
Template dei prompt e funzioni che costruiscono le catene LangChain
per i due passi del pipeline NER.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.output_parsers import JsonOutputParser

from .schemas import Paso1Salida, Paso2Salida
from .ontologia import CargadorOntologia


# ── Paso 1: Detección de spans ────────────────────────────────────────────────

SISTEMA_PASO1 = """\
Eres un sistema NLP especializado en identificar spans de texto
que corresponden a entidades nominales en documentos universitarios españoles.

Tu tarea es ÚNICAMENTE identificar los candidatos textuales (spans),
SIN clasificarlos todavía. Extrae cualquier fragmento que pueda ser:
- un programa, título o grado académico
- una normativa, reglamento o procedimiento
- un proceso de admisión, acceso o matrícula
- una unidad organizativa (universidad, comisión, centro)
- un documento institucional
- un requisito o condición
- un precio o tasa
- un término de calendario o planificación

{format_instructions}
"""

HUMANO_PASO1 = """\
Analiza el siguiente texto e identifica todos los spans candidatos.
Los índices start/end están basados en los caracteres del texto original.

TEXTO:
{text}
"""


def construir_cadena_paso1(llm: ChatOpenAI):
    """Devuelve la cadena LangChain para el Paso 1 (detección de spans)."""
    parser = JsonOutputParser(pydantic_object=Paso1Salida)
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SISTEMA_PASO1),
        HumanMessagePromptTemplate.from_template(HUMANO_PASO1),
    ]).partial(format_instructions=parser.get_format_instructions())
    return prompt | llm | parser


# ── Paso 2: Clasificación + Verificación ─────────────────────────────────────

SISTEMA_PASO2 = """\
Eres un sistema NER que clasifica spans de texto usando
una ontología del dominio universitario español.

=== ONTOLOGÍA: TIPOS DE ENTIDAD ===
{descripcion_entidades}

=== ONTOLOGÍA: RELACIONES ===
{descripcion_relaciones}

=== RESTRICCIONES ===
{restricciones}

Para cada span proporcionado:
1. Asigna el tipo de entidad más apropiado de la ontología (o "NONE" si no aplica)
2. Indica la confianza: "high" | "medium" | "low"
3. Identifica las relaciones entre entidades según la ontología

{format_instructions}
"""

HUMANO_PASO2 = """\
Clasifica los siguientes spans extraídos del texto.

TEXTO ORIGINAL:
{text}

SPANS A CLASIFICAR:
{spans_json}
"""


def construir_cadena_paso2(llm: ChatOpenAI, ontologia: CargadorOntologia):
    """Devuelve la cadena LangChain para el Paso 2 (clasificación)."""
    parser = JsonOutputParser(pydantic_object=Paso2Salida)
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SISTEMA_PASO2),
        HumanMessagePromptTemplate.from_template(HUMANO_PASO2),
    ]).partial(
        descripcion_entidades=ontologia.descripcion_entidades(),
        descripcion_relaciones=ontologia.descripcion_relaciones(),
        restricciones=ontologia.texto_restricciones(),
        format_instructions=parser.get_format_instructions(),
    )
    return prompt | llm | parser