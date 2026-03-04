"""
pipeline.py
-----------
Pipeline NER di due passi con verifica ontologica.

  Paso 1  — Detección de spans   (sin tipo)
  Paso 2  — Clasificación        (guiada por ontología)
  Paso 2b — Verificación         (comprobación de restricciones)

L'ontologia può essere fornita in due modi:
  A) YAML existente      → PipelineNERDosPasos(ruta_ontologia="ontology.yaml")
  B) Descubrimiento auto → PipelineNERDosPasos(ruta_vocabulario=Path("doc.md"))
                           Genera data/ner/ontologia/borrador_ontologia.json
                           que debe revisarse para crear el YAML definitivo.
"""

import json
from pathlib import Path
from typing import Optional

from langchain_openai import ChatOpenAI

from .ontologia import CargadorOntologia, HybridDocumentAnalyzer
from .verificador import VerificadorRestricciones
from .schemas import RelacionItem
from .prompts import construir_cadena_paso1, construir_cadena_paso2

_RUTA_BORRADOR = Path("data/ner/ontologia/borrador_ontologia.json")


class PipelineNERDosPasos:
    """
    Pipeline NER de dos pasos con verificación ontológica.
    Basada en LangChain + OpenAI GPT-4o.
    """

    def __init__(
        self,
        # ── Fuente de ontología (una de las dos) ──────────────────────────
        ruta_ontologia:   Optional[str]  = None,  # A) YAML estructurado
        ruta_vocabulario: Optional[Path] = None,  # B) doc para descubrimiento
        # ── Parámetros del analizador (modo B) ────────────────────────────
        top_n:                int   = 15,
        min_frecuencia:       int   = 2,
        threshold_percentual: float = 0.05,
        # ── LLM ──────────────────────────────────────────────────────────
        modelo:      str   = "gpt-4o",
        temperatura: float = 0.0,
    ):
        # ── Modo A: YAML ──────────────────────────────────────────────────
        if ruta_ontologia:
            self.ontologia = CargadorOntologia(ruta_ontologia)

        # ── Modo B: descubrimiento automático ─────────────────────────────
        elif ruta_vocabulario:
            print(f"🔍 Descubriendo vocabulario desde: {Path(ruta_vocabulario).name}")
            analizador = HybridDocumentAnalyzer(
                modelo_spacy="es_core_news_lg",
                patron_tabla=["Fila de tabla", "Tabla"],
            )
            resultado = analizador.analiza(
                Path(ruta_vocabulario),
                top_n=top_n,
                min_frecuencia=min_frecuencia,
                threshold_percentual=threshold_percentual,
                verbose=True,
            )

            # Guardar borrador para revisión manual
            _RUTA_BORRADOR.parent.mkdir(parents=True, exist_ok=True)
            analizador.guardar_ontology_json(
                ruta_salida=_RUTA_BORRADOR,
                nodos=resultado["allowed_nodes"],
                relaciones=resultado["allowed_relationships"],
                archivo_fuente=str(ruta_vocabulario),
            )
            print()
            print("⚠️  ATENCIÓN: se ha generado un borrador de ontología en:")
            print(f"   📄 {_RUTA_BORRADOR}")
            print("   Revísalo y conviértelo en un YAML estructurado con")
            print("   entities/relations/constraints antes de usarlo en producción.")
            print("   Luego pasa ruta_ontologia='ruta/al/ontology.yaml' al pipeline.")
            print()

            self.ontologia = _OntologiaMinimal(
                nodes=resultado["allowed_nodes"],
                relationships=resultado["allowed_relationships"],
            )
            print(f"✅ Vocabulario cargado: {len(resultado['allowed_nodes'])} nodos, "
                  f"{len(resultado['allowed_relationships'])} relaciones")

        else:
            raise ValueError(
                "Proporciona ruta_ontologia (YAML definitivo) "
                "o ruta_vocabulario (documento para descubrir vocabulario)."
            )

        self.llm          = ChatOpenAI(model=modelo, temperature=temperatura)
        self.cadena_paso1 = construir_cadena_paso1(self.llm)
        self.cadena_paso2 = construir_cadena_paso2(self.llm, self.ontologia)

    # ── Paso 1 ───────────────────────────────────────────────────────────────

    def ejecutar_paso1(self, texto: str) -> list[dict]:
        """Detección de spans sin tipo. Devuelve lista de {text, start, end}."""
        print("  [Paso 1] Detección de spans...")
        resultado = self.cadena_paso1.invoke({"text": texto})
        spans     = resultado.get("spans", []) if isinstance(resultado, dict) else []
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

        entidades_raw  = resultado.get("entidades",  []) if isinstance(resultado, dict) else []
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


# ─────────────────────────────────────────────────────────────────────────────
# Adattatore interno per il modo B
# ─────────────────────────────────────────────────────────────────────────────

class _OntologiaMinimal:
    """
    Implementa la stessa interfaccia di CargadorOntologia
    da liste di nodi/relazioni scoperte da HybridDocumentAnalyzer.
    Senza domain/range il VerificadorRestricciones accetta tutte le relazioni.
    """

    def __init__(self, nodes: list[str], relationships: list[str]):
        self._nodes         = nodes
        self._relationships = relationships

    def descripcion_entidades(self) -> str:
        return "\n".join(f"- **{n}**" for n in self._nodes)

    def descripcion_relaciones(self) -> str:
        return "\n".join(f"- **{r}**" for r in self._relationships)

    def texto_restricciones(self) -> str:
        return ""

    def tipos_entidad_validos(self) -> list[str]:
        return list(self._nodes)

    def restricciones_relaciones(self) -> dict:
        return {r: {"domain": self._nodes, "range": self._nodes}
                for r in self._relationships}