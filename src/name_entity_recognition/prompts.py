"""
prompts.py
----------
Plantillas de prompt y funciones que construyen las cadenas LangChain
para los dos pasos del pipeline NER.
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


# ─────────────────────────────────────────────────────────────────────────────
# PASO 1: Detección de spans
# ─────────────────────────────────────────────────────────────────────────────

SISTEMA_PASO1 = """\
Eres un sistema NLP especializado en identificar spans de texto
que corresponden a entidades nominales en documentos universitarios españoles.

Tu tarea es EXCLUSIVAMENTE identificar los candidatos textuales (spans),
SIN clasificarlos todavía. Extrae cualquier fragmento que pueda ser:

- un programa, título o grado académico
- una normativa, reglamento o procedimiento
- un proceso de admisión, acceso o matrícula
- una unidad organizativa (universidad, comisión, centro)
- un documento institucional
- un requisito o condición
- un precio o tasa
- un término de calendario o planificación

Reglas adicionales:
- No generes spans demasiado largos si pueden dividirse en unidades más precisas.
- Evita spans redundantes o solapados salvo que sea estrictamente necesario.
- Utiliza los índices de caracteres exactos del texto original.

{format_instructions}
"""

HUMANO_PASO1 = """\
Analiza el siguiente texto e identifica todos los spans candidatos.
Los índices start/end deben corresponder exactamente a las posiciones
del texto original.

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


# ─────────────────────────────────────────────────────────────────────────────
# PASO 2: Clasificación + Extracción de relaciones
# ─────────────────────────────────────────────────────────────────────────────

SISTEMA_PASO2 = """\
Eres un sistema NER que clasifica spans de texto y extrae relaciones
utilizando estrictamente la ontología del dominio universitario español.

=== ONTOLOGÍA: TIPOS DE ENTIDAD ===
{descripcion_entidades}

=== ONTOLOGÍA: RELACIONES ===
{descripcion_relaciones}

=== RESTRICCIONES ===
{restricciones}

=== INSTRUCCIONES ===

PASO A — Clasificación de entidades:
Para cada span proporcionado:

1. Asigna el tipo de entidad más apropiado ENTRE LOS DEFINIDOS EN LA ONTOLOGÍA.
2. Si ningún tipo coincide adecuadamente, asigna exactamente "NONE".
3. NO inventes entidades nuevas. SOLO puedes usar los tipos que aparecen
   en la ontología. No propongas categorías alternativas ni variantes.
4. Indica la confianza: "high" | "medium" | "low".
5. No alteres el texto del span. Utiliza el texto EXACTO en todos los campos.

PASO B — Extracción de relaciones:
Examina TODAS las combinaciones posibles de pares de entidades clasificadas y,
para cada par, decide si existe una relación definida en la ontología.

Reglas estrictas:

- Usa EXCLUSIVAMENTE los textos exactos de los spans como "sujeto" y "objeto".
- Una relación solo puede extraerse si aparece en la ontología.
- NO inventes relaciones. Deben estar explícita o implícitamente justificadas
  por el texto original.
- NO generes relaciones cuyo dominio o rango contradigan las restricciones de la ontología.
- Si una entidad no tiene ninguna relación válida, simplemente no generes ninguna.
- Extrae TODAS las relaciones presentes en el texto, pero NO añadas relaciones
  no justificadas.

{format_instructions}
"""

HUMANO_PASO2 = """\
Clasifica los siguientes spans y extrae todas las relaciones entre ellos
siguiendo estrictamente la ontología proporcionada.

TEXTO ORIGINAL:
{text}

SPANS A CLASIFICAR (usa estos textos EXACTOS para sujeto/objeto):
{spans_json}

Recuerda:
- Usa únicamente los tipos definidos en la ontología.
- Examina cada par de entidades.
- Extrae relaciones solo si están justificadas por el texto.
"""


def construir_cadena_paso2(llm: ChatOpenAI, ontologia: CargadorOntologia):
    """Devuelve la cadena LangChain para el Paso 2 (clasificación y relaciones)."""
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