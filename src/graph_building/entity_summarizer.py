"""
entity_summarizer.py
--------------------
Fase B del pipeline GraphRAG: consolidación de descripciones de entidades.

Problema que resuelve:
  Cada CanonicalEntity tiene una lista de descripciones parciales recogidas
  chunk a chunk por la pipeline NER. Por ejemplo, "Universidad de La Laguna"
  puede tener 8 descripciones distintas, una por cada chunk donde aparece.
  Este módulo las consolida en una única descripcion_consolidada usando un LLM,
  y luego la escribe en Neo4j sobre el nodo correspondiente.

Pipeline:
  1. Itera sobre todos los CanonicalEntity del EntityMap
  2. Si la entidad tiene 0-1 descripciones, no llama al LLM (no hace falta)
  3. Si tiene ≥2 descripciones, llama al LLM para consolidarlas
  4. Guarda descripcion_consolidada en el CanonicalEntity (en memoria)
  5. Escribe la descripción consolidada en Neo4j (SET e.descripcion_consolidada)

Uso típico (después de entity_resolver.py y graph_builder.py):
    summarizer = EntitySummarizer(graph=builder.graph, verbose=True)
    summarizer.summarize(entity_map)          # Fase B1: entidades
    summarizer.summarize_relations()          # Fase B2: relaciones
"""

import os
import time
from typing import Dict, List, Optional

from openai import OpenAI
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph

from .entity_resolver import CanonicalEntity, EntityMap

load_dotenv()

# Modelo a usar para la summarización
_MODEL = "gpt-4o"

# Número máximo de descripciones a enviar al LLM en una sola llamada.
# Si una entidad tiene más, se trunca (las más frecuentes primero).
_MAX_DESCRIPTIONS = 10

# Pausa entre llamadas al LLM para evitar rate limiting (segundos)
_SLEEP_BETWEEN_CALLS = 0.3


class EntitySummarizer:
    """
    Consolida las descripciones de cada CanonicalEntity con un LLM
    y actualiza los nodos en Neo4j.
    """

    def __init__(
        self,
        graph: Neo4jGraph,
        verbose: bool = True,
        max_descriptions: int = _MAX_DESCRIPTIONS,
        sleep_between_calls: float = _SLEEP_BETWEEN_CALLS,
    ):
        """
        Args:
            graph:                Conexión Neo4j activa (la misma de GraphBuilder).
            verbose:              Log de progreso.
            max_descriptions:     Máximo de descripciones por llamada LLM.
            sleep_between_calls:  Pausa entre llamadas para evitar rate limiting.
        """
        self.graph                = graph
        self.verbose              = verbose
        self.max_descriptions     = max_descriptions
        self.sleep_between_calls  = sleep_between_calls
        self.client               = OpenAI()

    # ── Punto de entrada ──────────────────────────────────────────────────────

    def summarize(self, entity_map: EntityMap) -> Dict[str, int]:
        """
        Procesa todos los CanonicalEntity del mapa.

        Returns:
            Dict con contadores: {"procesadas": N, "con_llm": N, "sin_llm": N, "errores": N}
        """
        canonicals = entity_map.all_canonicals()
        total      = len(canonicals)

        stats = {"procesadas": 0, "con_llm": 0, "sin_llm": 0, "errores": 0}

        print(f"\n📝 EntitySummarizer — {total} entidades canónicas a procesar...")

        for i, ce in enumerate(canonicals, 1):
            if self.verbose:
                print(f"   [{i}/{total}] [{ce.entity_type}] '{ce.canonical_text}' "
                      f"({len(ce.descriptions)} descripciones)", end=" ")

            try:
                if len(ce.descriptions) <= 1:
                    # No hace falta LLM: copia la descripción existente o deja vacío
                    ce.descripcion_consolidada = ce.descriptions[0] if ce.descriptions else ""
                    stats["sin_llm"] += 1
                    if self.verbose:
                        print("→ sin LLM")
                else:
                    # Llama al LLM para consolidar
                    ce.descripcion_consolidada = self._consolidar(ce)
                    stats["con_llm"] += 1
                    if self.verbose:
                        print("→ consolidada")
                    time.sleep(self.sleep_between_calls)

                # Escribe en Neo4j
                self._escribir_en_neo4j(ce)
                stats["procesadas"] += 1

            except Exception as e:
                stats["errores"] += 1
                if self.verbose:
                    print(f"→ ⚠️ error: {e}")

        print(f"\n✅ Summarización completada:")
        print(f"   • Con LLM    : {stats['con_llm']}")
        print(f"   • Sin LLM    : {stats['sin_llm']}")
        print(f"   • Errores    : {stats['errores']}")
        print(f"   • Total      : {stats['procesadas']}/{total}")

        return stats

    # ── Consolidación con LLM ─────────────────────────────────────────────────

    def _consolidar(self, ce: CanonicalEntity) -> str:
        """
        Llama al LLM para consolidar las descripciones de una CanonicalEntity.
        Trunca a max_descriptions para no superar el contexto.
        """
        # Toma las descripciones más informativas (no vacías, sin duplicados)
        descripciones = list(dict.fromkeys(d for d in ce.descriptions if d.strip()))
        descripciones = descripciones[: self.max_descriptions]

        if not descripciones:
            return ""

        # Si solo queda una después de deduplicar, no hace falta LLM
        if len(descripciones) == 1:
            return descripciones[0]

        prompt = self._build_prompt(ce, descripciones)

        response = self.client.chat.completions.create(
            model=_MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()

    def _build_prompt(self, ce: CanonicalEntity, descripciones: List[str]) -> str:
        """
        Construye el prompt para la consolidación.
        El prompt es deliberadamente breve para maximizar velocidad con Haiku.
        """
        lista = "\n".join(f"- {d}" for d in descripciones)

        return f"""Eres un asistente especializado en normativa universitaria española.
Se te dan {len(descripciones)} descripciones de la entidad "{ce.canonical_text}" \
(tipo: {ce.entity_type}), extraídas de distintos fragmentos de un reglamento de la \
Universidad de La Laguna.

Genera una descripción consolidada: precisa, concisa (máximo 2 frases) y en español.
No repitas el nombre de la entidad al inicio.

Descripciones:
{lista}

Responde SOLO con la descripción consolidada, sin preámbulo ni explicación."""

    # ── Escritura en Neo4j ────────────────────────────────────────────────────

    def _escribir_en_neo4j(self, ce: CanonicalEntity):
        """
        Actualiza el nodo Entidad en Neo4j con la descripción consolidada.
        Usa el canonical_id como clave para el MATCH (univoco, sin ambigüedad).
        """
        self.graph.query(
            """
            MATCH (e:Entidad {id: $id})
            SET e.descripcion_consolidada = $descripcion
            """,
            {
                "id":          ce.canonical_id,
                "descripcion": ce.descripcion_consolidada,
            },
        )

    # ── Fase B2: summarización de relaciones ──────────────────────────────────

    def summarize_relations(self) -> Dict[str, int]:
        """
        Consolida las descripciones de cada relación del grafo.

        Para cada arco (s)-[:RELACION]->(o), lee los textos de los chunks
        fuente donde aparece esa relación y genera una descripción consolidada.
        La escribe como propiedad `descripcion` en el arco.

        Returns:
            Dict con contadores: {"procesadas": N, "con_llm": N, "sin_llm": N, "errores": N}
        """
        # Lee todas las relaciones del grafo junto con los textos de sus chunks fuente.
        # Excluye MENTIONED_IN y BELONGS_TO que son arcos estructurales, no semánticos.
        relaciones = self.graph.query(
            """
            MATCH (s:Entidad)-[r]->(o:Entidad)
            WHERE type(r) <> 'MENTIONED_IN' AND type(r) <> 'BELONGS_TO'
            WITH s, r, o,
                 [(s)-[:MENTIONED_IN]->(c:Chunk) | c.texto][0..5] AS textos_fuente
            RETURN
                id(r)        AS rel_id,
                type(r)      AS rel_type,
                s.text       AS sujeto,
                s.entity_type AS tipo_sujeto,
                o.text       AS objeto,
                o.entity_type AS tipo_objeto,
                textos_fuente
            """
        )

        total = len(relaciones)
        stats = {"procesadas": 0, "con_llm": 0, "sin_llm": 0, "errores": 0}

        print(f"\n📝 EntitySummarizer (relaciones) — {total} relaciones a procesar...")

        for i, rel in enumerate(relaciones, 1):
            if self.verbose:
                print(f"   [{i}/{total}] ({rel['sujeto']})-[:{rel['rel_type']}]"
                      f"->({rel['objeto']})", end=" ")
            try:
                textos = [t for t in rel.get("textos_fuente", []) if t and t.strip()]

                if not textos:
                    # Sin contexto textual: genera descripción mínima sin LLM
                    descripcion = (f"Relación {rel['rel_type']} entre "
                                   f"{rel['sujeto']} y {rel['objeto']}.")
                    stats["sin_llm"] += 1
                    if self.verbose:
                        print("→ sin contexto")
                else:
                    descripcion = self._consolidar_relacion(rel, textos)
                    stats["con_llm"] += 1
                    if self.verbose:
                        print("→ consolidada")
                    time.sleep(self.sleep_between_calls)

                self._escribir_relacion_en_neo4j(rel["rel_id"], descripcion)
                stats["procesadas"] += 1

            except Exception as e:
                stats["errores"] += 1
                if self.verbose:
                    print(f"→ ⚠️ error: {e}")

        print(f"\n✅ Summarización de relaciones completada:")
        print(f"   • Con LLM    : {stats['con_llm']}")
        print(f"   • Sin LLM    : {stats['sin_llm']}")
        print(f"   • Errores    : {stats['errores']}")
        print(f"   • Total      : {stats['procesadas']}/{total}")

        return stats

    def _consolidar_relacion(self, rel: Dict, textos: List[str]) -> str:
        """
        Genera una descripción de la relación basándose en los textos fuente
        donde aparecen ambas entidades relacionadas.
        """
        # Trunca los textos para no superar el contexto
        textos_prompt = textos[: self.max_descriptions]
        fragmentos    = "\n\n".join(f"[Fragmento {i+1}]: {t[:400]}"
                                    for i, t in enumerate(textos_prompt))

        prompt = self._build_relation_prompt(rel, fragmentos)

        response = self.client.chat.completions.create(
            model=_MODEL,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()

    def _build_relation_prompt(self, rel: Dict, fragmentos: str) -> str:
        return f"""Eres un asistente especializado en normativa universitaria española.
En un reglamento de la Universidad de La Laguna existe la siguiente relación:

  [{rel['tipo_sujeto']}] "{rel['sujeto']}"  -[{rel['rel_type']}]→  [{rel['tipo_objeto']}] "{rel['objeto']}"

Estos son fragmentos del reglamento donde aparecen ambas entidades:
{fragmentos}

Genera una descripción concisa (máximo 2 frases) que explique qué significa
esta relación en el contexto de la normativa universitaria.
Responde SOLO con la descripción, en español, sin preámbulo."""

    def _escribir_relacion_en_neo4j(self, rel_id: int, descripcion: str):
        """
        Escribe la descripción consolidada en la propiedad `descripcion` del arco.
        Usa el id interno de Neo4j para identificar el arco de forma unívoca.
        """
        self.graph.query(
            """
            MATCH ()-[r]->()
            WHERE id(r) = $rel_id
            SET r.descripcion = $descripcion
            """,
            {"rel_id": rel_id, "descripcion": descripcion},
        )


# ── Uso típico ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from .entity_resolver import EntityResolver
    from .graph_builder import GraphBuilder

    ruta_json = sys.argv[1] if len(sys.argv) > 1 else "ner_resultados.json"

    # Fase A: resolución de entidades
    resolver   = EntityResolver(fuzzy_threshold=0.93, verbose=True)
    entity_map = resolver.resolve(ruta_json)
    entity_map.print_stats()

    # Construcción del grafo
    builder = GraphBuilder(
        verbose=True,
        solo_relaciones_validas=True,
        confianza_minima="medium",
        entity_map=entity_map,
    )
    builder.construir_desde_json(ruta_json, limpiar=True)

    # Fase B1: summarización de entidades
    summarizer = EntitySummarizer(graph=builder.graph, verbose=True)
    summarizer.summarize(entity_map)

    # Fase B2: summarización de relaciones
    summarizer.summarize_relations()

    builder.estadisticas()