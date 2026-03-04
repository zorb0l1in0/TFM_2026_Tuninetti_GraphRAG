"""
pipeline.py
-----------
Pipeline NER di due passi con verifica ontologica.

  Paso 1  — Detección de spans   (sin tipo)
  Paso 2  — Clasificación        (guiada por ontología)
  Paso 2b — Verificación         (comprobación de restricciones)
"""

import json

from langchain_openai import ChatOpenAI

from .ontologia import CargadorOntologia
from .verificador import VerificadorRestricciones
from .schemas import RelacionItem
from .prompts import construir_cadena_paso1, construir_cadena_paso2


class PipelineNERDosPasos:
    """
    Pipeline NER de dos pasos con verificación ontológica.
    Basada en LangChain + OpenAI GPT-4o.
    """

    def __init__(
        self,
        ruta_ontologia: str,
        modelo: str = "gpt-4o",
        temperatura: float = 0.0,
    ):
        self.ontologia    = CargadorOntologia(ruta_ontologia)
        self.llm          = ChatOpenAI(model=modelo, temperature=temperatura)
        self.cadena_paso1 = construir_cadena_paso1(self.llm)
        self.cadena_paso2 = construir_cadena_paso2(self.llm, self.ontologia)

    # ── Paso 1 ───────────────────────────────────────────────────────────────

    def ejecutar_paso1(self, texto: str) -> list[dict]:
        """Detección de spans sin tipo. Devuelve lista de {text, start, end}."""
        print("  [Paso 1] Detección de spans...")
        resultado = self.cadena_paso1.invoke({"text": texto})
        spans = resultado.get("spans", []) if isinstance(resultado, dict) else []
        print(f"  [Paso 1] {len(spans)} spans candidatos encontrados.")
        return spans

    # ── Paso 2 ───────────────────────────────────────────────────────────────

    def ejecutar_paso2(
        self, texto: str, spans: list[dict]
    ) -> tuple[list[dict], list[dict]]:
        """Clasificación guiada por ontología + verificación de restricciones."""
        print("  [Paso 2] Clasificación + verificación ontológica...")

        spans_json = json.dumps(spans, ensure_ascii=False, indent=2)
        resultado  = self.cadena_paso2.invoke({"text": texto, "spans_json": spans_json})

        entidades_raw  = resultado.get("entidades", [])  if isinstance(resultado, dict) else []
        relaciones_raw = resultado.get("relaciones", []) if isinstance(resultado, dict) else []

        # Filtrar entidades NONE y tipos no válidos
        tipos_validos = self.ontologia.tipos_entidad_validos()
        entidades = [
            e for e in entidades_raw
            if isinstance(e, dict)
            and e.get("entity_type") not in ("NONE", None)
            and e.get("entity_type") in tipos_validos
        ]

        # Mapa texto → tipo para el verificador
        mapa_entidades = {e["text"]: e["entity_type"] for e in entidades}

        # Verificación de restricciones
        items_relacion = [
            RelacionItem(
                sujeto=r.get("sujeto", ""),
                relacion=r.get("relacion", ""),
                objeto=r.get("objeto", ""),
            )
            for r in relaciones_raw if isinstance(r, dict)
        ]
        verificador = VerificadorRestricciones(self.ontologia, mapa_entidades)
        relaciones  = verificador.verificar(items_relacion)

        validas = sum(1 for r in relaciones if r["valida"])
        print(f"  [Paso 2] {len(entidades)} entidades, {validas}/{len(relaciones)} relaciones válidas.")
        return entidades, relaciones

    # ── Pipeline completa ─────────────────────────────────────────────────────

    def ejecutar(self, texto: str) -> dict:
        print(f"\n{'='*60}")
        print(f"Procesando ({len(texto)} caracteres)...")
        print(f"{'='*60}")

        spans = self.ejecutar_paso1(texto)
        if not spans:
            return {"texto": texto, "entidades": [], "relaciones": [], "spans_paso1": []}

        entidades, relaciones = self.ejecutar_paso2(texto, spans)
        return {
            "texto":       texto,
            "spans_paso1": spans,
            "entidades":   entidades,
            "relaciones":  relaciones,
        }

    # ── Formato legible ───────────────────────────────────────────────────────

    def formatear_resultado(self, resultado: dict) -> str:
        lineas = [
            f"\n{'='*60}",
            "RESULTADOS NER",
            f"{'='*60}",
            f"\nTEXTO: {resultado['texto'][:120]}{'...' if len(resultado['texto']) > 120 else ''}",
            f"\n{'─'*40}",
            f"PASO 1 — SPANS CANDIDATOS ({len(resultado.get('spans_paso1', []))}):",
        ]
        for s in resultado.get("spans_paso1", []):
            lineas.append(f"  • \"{s.get('text', '')}\" [{s.get('start', 0)}:{s.get('end', 0)}]")

        lineas += [
            f"\n{'─'*40}",
            f"PASO 2 — ENTIDADES CLASIFICADAS ({len(resultado.get('entidades', []))}):",
        ]
        for e in resultado.get("entidades", []):
            icono = {"high": "✓", "medium": "~", "low": "?"}.get(e.get("confidence", ""), "?")
            lineas.append(f"  {icono} [{e.get('entity_type')}] \"{e.get('text')}\"")
            if e.get("descripcion"):
                lineas.append(f"      → {e['descripcion']}")

        lineas += [
            f"\n{'─'*40}",
            f"PASO 2b — RELACIONES + VERIFICACIÓN ({len(resultado.get('relaciones', []))}):",
        ]
        for r in resultado.get("relaciones", []):
            ok    = "✓" if r.get("valida") else "✗"
            linea = f"  {ok} {r.get('sujeto')} --[{r.get('relacion')}]--> {r.get('objeto')}"
            if not r.get("valida") and r.get("motivo"):
                linea += f"\n      ⚠ {r['motivo']}"
            lineas.append(linea)

        return "\n".join(lineas)