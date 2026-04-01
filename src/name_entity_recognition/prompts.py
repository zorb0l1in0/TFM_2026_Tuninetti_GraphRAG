"""
prompts.py
----------
Plantillas de prompt y cadenas LangChain para el pipeline NER de dos pasos.

FIXES:
  1. ThinkStripParser — maneja <think> cerrado, truncado o ausente,
     Y también modelos que razonan en texto libre antes del JSON final.
  2. construir_cadena_paso1/paso2 reciben LLMs separados:
       - Paso 1: razonamiento habilitado (presupuesto bajo)
       - Paso 2: razonamiento DESHABILITADO (sin <think>)
  3. Escape de llaves JSON en los templates para evitar KeyError
     'Input to ChatPromptTemplate is missing variables {"spans"}'.
"""

import json
import re

from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers.base import BaseOutputParser

from .schemas import Paso2Salida
from .ontologia import CargadorOntologia




PROMPT_GENERAR_ONTOLOGIA = """
/no_think
Eres un experto en modelado de ontologías académicas.

Debes generar un archivo ontology.yaml COMPLETO y ESTRICTAMENTE válido, a partir de un JSON que contiene:
- allowed_nodes
- allowed_relationships

FORMATO OBLIGATORIO (MAPPING — compatible con yaml.safe_load):

entities:
  NombreEntidad:
    description: frase breve y clara
    patterns:
      - patron1
      - patron2

relations:
  NOMBRE_RELACION:
    description: frase breve
    domain:
      - Entidad1
    range:
      - Entidad1
      - Entidad2

constraints:
  - domain y range SIEMPRE deben ser listas (aunque haya uno solo).
  - No relaciones reflexivas salvo RELACIONAR.
  - Las entidades de fecha (meses, años) son modificadores: NO aparecen en domain ni range.
  - patterns SIEMPRE en minúsculas y sin comillas dobles.
  - No añadir explicación adicional fuera del YAML.
  - YAML puro con indentación de 2 espacios.
  - Usa EXCLUSIVAMENTE los tipos de entidad definidos en 'entities'. Si ninguno encaja, usa NONE.
  - Usa EXCLUSIVAMENTE las relaciones definidas en 'relations'. Está PROHIBIDO inventar relaciones nuevas como MODIFICAR, DELEGAR, GESTIONAR o similares.
  - El sujeto de cada relación debe pertenecer al dominio (domain) de esa relación.
  - El objeto de cada relación debe pertenecer al rango (range) de esa relación.
  - Si no existe ninguna relación válida entre dos entidades, no generes ninguna.
  - Usa el texto EXACTO del span como sujeto y objeto, sin parafrasear.


IMPORTANTE:
- Usa EXCLUSIVAMENTE el formato mapping ('NombreEntidad:') tanto para entities como para relations.
- NO uses listas con '- name:' porque son incompatibles con el parser.
- Normaliza nombres a Mayúscula Inicial para entidades, MAYÚSCULAS para relaciones.
- No incluyas backticks ni bloques ```yaml```.

Ahora genera ÚNICAMENTE el archivo ontology.yaml FINAL.

"""


# ─────────────────────────────────────────────────────────────────────────────
# Utilidad: extrae el último JSON válido de una cadena de texto
# ─────────────────────────────────────────────────────────────────────────────

_CLAVES_ESPERADAS = {"spans", "entidades", "relaciones"}

def _extract_last_json(text: str) -> str:
    # 0. Elimina el bloque <think>...</think> explícito (modelos Qwen3 con vLLM)
    if "</think>" in text:
        text = text.split("</think>", 1)[-1].strip()

    # NOTA: se eliminó el bloque elif "Thinking Process:" que causaba el bug:
    # text.find("{") encontraba JSONs parciales dentro del razonamiento libre
    # del modelo, en lugar del JSON final correcto al final de la respuesta.

    # 1. Busca bloques markdown ```json ... ``` o ``` ... ```
    md_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text)
    # Primero busca el último bloque que contenga claves conocidas del pipeline
    for block in reversed(md_blocks):
        try:
            parsed = json.loads(block.strip())
            if isinstance(parsed, dict) and _CLAVES_ESPERADAS & parsed.keys():
                return block.strip()
        except json.JSONDecodeError:
            pass
    # Si no hay coincidencia con claves conocidas, devuelve el último bloque válido
    for block in reversed(md_blocks):
        try:
            json.loads(block.strip())
            return block.strip()
        except json.JSONDecodeError:
            pass

    # 2. Escanea todo el texto buscando candidatos JSON ({...} o [...])
    # Necesario cuando el modelo responde en texto libre sin bloques markdown
    candidates = []
    decoder = json.JSONDecoder()
    for match in re.finditer(r"[{\[]", text):
        start = match.start()
        try:
            obj, end_offset = decoder.raw_decode(text, start)
            raw = text[start:start + end_offset]
            candidates.append((start, raw, obj))
        except json.JSONDecodeError:
            continue

    if not candidates:
        raise ValueError(
            "No se encontró ningún JSON válido en la respuesta del modelo.\n"
            f"Respuesta recibida (primeros 500 chars):\n{text[:500]}..."
        )

    # Prefiere el último JSON que contenga claves esperadas (spans/entidades/relaciones)
    # El modelo siempre genera el JSON correcto AL FINAL del razonamiento,
    # por eso se usa max() en lugar de min() para obtener la posición más tardía
    known = [
        (start, raw) for start, raw, obj in candidates
        if isinstance(obj, dict) and _CLAVES_ESPERADAS & obj.keys()
    ]
    if known:
        _, best = max(known, key=lambda x: x[0])
        return best

    # Fallback: devuelve el último JSON encontrado si ninguno tiene claves conocidas
    _, best, _ = max(candidates, key=lambda x: x[0])
    return best


# ─────────────────────────────────────────────────────────────────────────────
# ThinkStripParser
# ─────────────────────────────────────────────────────────────────────────────

class ThinkStripParser(BaseOutputParser):
    """
    Wrapper sobre JsonOutputParser que:
    1. Elimina bloques <think>...</think> (completos o truncados).
    2. Extrae el ÚLTIMO JSON válido del texto, ignorando razonamiento libre
       que el modelo pueda escribir antes del JSON final.
    """

    inner_parser: JsonOutputParser

    class Config:
        arbitrary_types_allowed = True

    def parse(self, text: str) -> dict:
        # ── Paso 1: eliminar bloque <think>...</think> completo ──────────────
        clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

        # ── Paso 2: bloque <think> sin cerrar (max_tokens agotado) ───────────
        if "<think>" in clean:
            after = re.split(r"<think>", clean, maxsplit=1)[-1]
            try:
                return self.inner_parser.parse(_extract_last_json(after))
            except ValueError:
                raise ValueError(
                    "El modelo agotó los tokens dentro de <think> sin generar JSON.\n"
                    "Solución: aumenta max_tokens en llm_client.py o usa "
                    "enable_thinking=False para el Paso 2.\n\n"
                    f"Respuesta recibida (primeros 400 chars):\n{text[:400]}..."
                )

        # ── Paso 3: extraer el último JSON válido del texto limpio ───────────
        json_str = _extract_last_json(clean)
        return self.inner_parser.parse(json_str)

    def get_format_instructions(self) -> str:
        return self.inner_parser.get_format_instructions()

    @property
    def _type(self) -> str:
        return "think_strip_parser"


# ─────────────────────────────────────────────────────────────────────────────
# PASO 1 — Detección de spans
# ─────────────────────────────────────────────────────────────────────────────

# NOTA: las llaves literales en el ejemplo JSON deben escaparse con {{}}
# para que ChatPromptTemplate no las interprete como variables del template.

SISTEMA_PASO1 = """\
/no_think
Eres un extractor de spans para un Knowledge Graph del dominio universitario español.

Tu tarea es identificar las menciones de entidades relevantes para construir un grafo
de conocimiento. Cuando tengas dudas, INCLUYE el span.

EXTRAE:
1. Nombres propios: organizaciones, órganos, documentos normativos, titulaciones, cargos.
2. Roles de personas que actúan en el texto: cualquier mención a un tipo de persona
   con derechos u obligaciones en el contexto universitario
   (ej. "el estudiante", "el solicitante", "la persona interesada", "el deportista").
3. Conceptos procesales: procedimientos, recursos, expedientes, solicitudes, convocatorias.
4. Referencias normativas con número: "Real Decreto 534/2024".

NO EXTRAE:
- Números de artículo: "artículo 29", "artículo 16.4"
- Fechas y cantidades: "14 de mayo", "6 créditos", "un mes"
- URLs y sedes web: "web de la ULL"
- Referencias a anexos: "Anexo I", "Anexo II"
- Localidades geográficas: "San Cristóbal de La Laguna"

FORMATO — SOLO JSON:
{{
  "spans": [
    {{"text": "mención exacta y limpia del texto"}}
  ]
}}
"""

HUMANO_PASO1 = """\
Extrae del siguiente texto SOLO las menciones de entidades nombradas relevantes
para un Knowledge Graph universitario. Aplica los criterios del sistema.
Responde ÚNICAMENTE con el JSON, sin explicaciones.

TEXTO:
{text}
"""


def construir_cadena_paso1(llm: ChatOpenAI):
    # Sin pydantic_object: evita que format_instructions complejo confunda al modelo
    inner  = JsonOutputParser()
    parser = ThinkStripParser(inner_parser=inner)
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SISTEMA_PASO1),
        HumanMessagePromptTemplate.from_template(HUMANO_PASO1),
    ])
    # No inyectamos format_instructions en el prompt: el sistema ya indica el formato exacto
    return prompt | llm | parser


# ─────────────────────────────────────────────────────────────────────────────
# PASO 2 — Clasificación + relaciones
# ─────────────────────────────────────────────────────────────────────────────

SISTEMA_PASO2 = """\
/no_think
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
3. NO inventes entidades nuevas. SOLO puedes usar los tipos que aparecen en la ontología.
4. Indica la confianza: "high" | "medium" | "low".
5. Utiliza el texto EXACTO del span en todos los campos.
6. Genera una descripción breve (máximo 1 frase) en español que explique qué es
   esa entidad en el contexto del texto. Campo "descripcion" obligatorio, nunca vacío.

PASO B — Extracción de relaciones:
- Usa EXCLUSIVAMENTE los textos exactos de los spans como "sujeto" y "objeto".
- PROHIBIDO usar como sujeto u objeto cualquier texto que no aparezca en la lista de spans proporcionada.
- Solo extrae relaciones definidas en la ontología y justificadas por el texto.
- NO generes relaciones cuyo dominio/rango contradigan las restricciones.
- Si una entidad no tiene relación válida, no generes ninguna para ella.

- NO escribas explicaciones, razonamientos ni texto adicional.
- Responde ÚNICAMENTE con el objeto JSON final. Nada más.

{format_instructions}
"""

HUMANO_PASO2 = """\
Clasifica los siguientes spans y extrae todas las relaciones entre ellos
siguiendo estrictamente la ontología proporcionada.
Responde SOLO con el JSON. Sin explicaciones.

TEXTO ORIGINAL:
{text}

SPANS A CLASIFICAR (usa estos textos EXACTOS para sujeto/objeto):
{spans_json}
"""


def construir_cadena_paso2(llm: ChatOpenAI, ontologia: CargadorOntologia):
    inner  = JsonOutputParser(pydantic_object=Paso2Salida)
    parser = ThinkStripParser(inner_parser=inner)
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SISTEMA_PASO2),
        HumanMessagePromptTemplate.from_template(HUMANO_PASO2),
    ]).partial(
        descripcion_entidades=ontologia.descripcion_entidades_compacta(),
        descripcion_relaciones=ontologia.descripcion_relaciones_compacta(),
        restricciones=ontologia.texto_restricciones(),
        format_instructions=parser.get_format_instructions(),
    )
    return prompt | llm | parser


# ─────────────────────────────────────────────────────────────────────────────
# SISTEMA_ONTOLOGIA (sin cambios)
# ─────────────────────────────────────────────────────────────────────────────

SISTEMA_ONTOLOGIA = """\
/no_think
Eres un knowledge engineer experto.
Analiza los siguientes documentos Markdown y genera un borrador de ontología \
para un Knowledge Graph en formato JSON.

La ontología debe contener:
- "entidades": lista de tipos de entidad (ej. Persona, Organización, Lugar, Concepto…)
  Cada tipo tiene: "nombre", "descripcion", "atributos" (lista de atributos clave)
- "relaciones": lista de relaciones entre entidades
  Cada relación tiene: "nombre", "origen", "destino", "descripcion"
- "notas": posibles notas o decisiones de diseño

Responde SOLO con JSON válido, sin texto adicional ni backticks.

DOCUMENTOS:
\"\"\"
{documentos}
\"\"\"
"""