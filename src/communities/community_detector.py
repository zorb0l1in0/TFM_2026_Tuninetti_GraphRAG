"""
community_detector.py
---------------------
Fase C del pipeline GraphRAG: detección de comunidades multi-nivel y summaries.

Problema que resuelve:
  El grafo tiene cientos de nodos Entidad conectados por relaciones semánticas.
  Sin organización jerárquica, una consulta global ("¿qué procesos regula la ULL?")
  requeriría recorrer todo el grafo. Las comunidades agrupan nodos densamente
  conectados en clústeres temáticos a múltiples niveles de granularidad:
    - Nivel 0 (fino):   comunidades pequeñas y específicas
    - Nivel 1 (medio):  agrupaciones de comunidades de nivel 0
    - Nivel 2 (grueso): temas macro que agrupan todo un dominio
  Un LLM genera un summary por cada comunidad en cada nivel.
  Estos summaries son el núcleo del método GraphRAG "From Local to Global".

Pipeline:
  1. Proyecta el grafo en memoria con Neo4j GDS
  2. Ejecuta Leiden con includeIntermediateCommunities: true
  3. Por cada nivel detectado, lee los miembros y genera summaries
  4. Guarda nodos :Community {level: N} en Neo4j con :BELONGS_TO y :PARENT_OF

Requisitos:
  - Neo4j GDS instalado (plugin neo4j-graph-data-science)
  - Variables de entorno: NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, OPENAI_API_KEY

Uso típico (después de entity_summarizer.py):
    detector = CommunityDetector(graph=builder.graph, verbose=True)
    detector.detect_and_summarize()
"""

import time
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from langchain_community.graphs import Neo4jGraph

_MODEL           = "gpt-4o"
_GDS_GRAPH_NAME  = "entidades_graph"
_SLEEP_BETWEEN_CALLS = 0.5
_MAX_NODOS_EN_PROMPT = 30
_MAX_LEVELS      = 3   # niveles Leiden a procesar (0, 1, 2)


class CommunityDetector:
    """
    Detecta comunidades multi-nivel con Leiden (via GDS) e genera
    un summary LLM per ogni comunità ad ogni livello.

    Struttura Neo4j risultante:
      (:Entidad)-[:BELONGS_TO]->(:Community {level: 0})
      (:Community {level: 0})-[:PARENT_OF]->(:Community {level: 1})
      (:Community {level: 1})-[:PARENT_OF]->(:Community {level: 2})
    """

    def __init__(
        self,
        graph: Neo4jGraph,
        verbose: bool = True,
        min_community_size: int = 2,
        sleep_between_calls: float = _SLEEP_BETWEEN_CALLS,
        max_nodos_en_prompt: int = _MAX_NODOS_EN_PROMPT,
        gds_graph_name: str = _GDS_GRAPH_NAME,
        max_levels: int = _MAX_LEVELS,
    ):
        """
        Args:
            graph:               Conexión Neo4j activa.
            verbose:             Log de progreso.
            min_community_size:  Comunidades con menos nodos se ignoran.
            sleep_between_calls: Pausa entre llamadas LLM.
            max_nodos_en_prompt: Trunca comunidades grandes antes del LLM.
            gds_graph_name:      Nombre del grafo proyectado en GDS.
            max_levels:          Número máximo de niveles Leiden a procesar.
        """
        self.graph               = graph
        self.verbose             = verbose
        self.min_community_size  = min_community_size
        self.sleep_between_calls = sleep_between_calls
        self.max_nodos_en_prompt = max_nodos_en_prompt
        self.gds_graph_name      = gds_graph_name
        self.max_levels          = max_levels
        self.client              = OpenAI()

    # ── Punto de entrada ──────────────────────────────────────────────────────

    def detect_and_summarize(self) -> Dict[str, Any]:
        """
        Ejecuta el pipeline multi-nivel completo:
        proyección GDS → Leiden multi-nivel → summaries por nivel → Neo4j.

        Returns:
            Dict con estadísticas por nivel: {0: {"detectadas": N, "summaries": N}, ...}
        """
        print("\n🔍 CommunityDetector — detección multi-nivel...")

        self._limpiar_estado_previo()

        n_nodos, n_arcos = self._proyectar_gds()
        print(f"   Grafo proyectado: {n_nodos} nodos, {n_arcos} arcos")

        # Ejecuta Leiden con comunidades intermedias y obtiene los niveles reales
        niveles_detectados = self._ejecutar_leiden_multinivel()
        n_niveles = min(len(niveles_detectados), self.max_levels)
        print(f"   Leiden detectó {len(niveles_detectados)} niveles "
              f"(procesando {n_niveles})")

        stats_por_nivel: Dict[int, Dict] = {}

        for level in range(n_niveles):
            print(f"\n   📊 Nivel {level} ({'fino' if level == 0 else 'medio' if level == 1 else 'grueso'}):")
            comunidades    = self._leer_comunidades_nivel(level)
            validas        = [c for c in comunidades
                              if len(c["miembros"]) >= self.min_community_size]
            print(f"      {len(validas)} comunidades válidas (≥{self.min_community_size} miembros)")

            summaries_ok = 0
            for i, comunidad in enumerate(validas, 1):
                cid = comunidad["community_id"]
                if self.verbose:
                    print(f"      [{i}/{len(validas)}] C{cid} "
                          f"({len(comunidad['miembros'])} miembros)", end=" ")
                try:
                    summary = self._generar_summary(comunidad, level)
                    self._guardar_community_node(cid, level, comunidad, summary)
                    summaries_ok += 1
                    if self.verbose:
                        print("→ ✅")
                    time.sleep(self.sleep_between_calls)
                except Exception as e:
                    if self.verbose:
                        print(f"→ ⚠️ {e}")

            stats_por_nivel[level] = {
                "detectadas": len(comunidades),
                "summaries":  summaries_ok,
            }

        # Conecta comunidades di livello N con quelle di livello N+1 (PARENT_OF)
        for level in range(n_niveles - 1):
            self._conectar_niveles(level, level + 1)

        self._eliminar_proyeccion_gds()

        print(f"\n✅ Community detection multi-nivel completada:")
        for level, s in stats_por_nivel.items():
            print(f"   • Nivel {level}: {s['detectadas']} comunidades, "
                  f"{s['summaries']} summaries")

        return stats_por_nivel

    # ── GDS: proyección y Leiden ───────────────────────────────────────────────

    def _limpiar_estado_previo(self):
        """Elimina proyecciones GDS anteriores y nodos Community del grafo."""
        self.graph.query("MATCH (c:Community) DETACH DELETE c")
        # Elimina tutte le proprietà communityId_level_N
        for level in range(self.max_levels):
            self.graph.query(f"MATCH (e:Entidad) REMOVE e.communityId_level_{level}")
        self._eliminar_proyeccion_gds(silent=True)
        if self.verbose:
            print("   Estado previo limpiado")

    def _proyectar_gds(self) -> Tuple[int, int]:
        """
        Proyecta nodos Entidad y relaciones semánticas en GDS (UNDIRECTED).
        Excluye MENTIONED_IN y BELONGS_TO que son arcos estructurales.
        """
        result = self.graph.query(
            f"""
            CALL gds.graph.project(
                '{self.gds_graph_name}',
                'Entidad',
                {{
                    __ALL_RELATIONSHIPS__: {{
                        orientation: 'UNDIRECTED',
                        properties: {{}}
                    }}
                }}
            )
            YIELD nodeCount, relationshipCount
            RETURN nodeCount, relationshipCount
            """
        )
        if not result:
            raise RuntimeError("GDS no devolvió resultado. "
                               "¿Está instalado neo4j-graph-data-science?")
        row = result[0]
        return row["nodeCount"], row["relationshipCount"]

    def _ejecutar_leiden_multinivel(self) -> List[int]:
        """
        Ejecuta Leiden con includeIntermediateCommunities: true.
        Escribe communityId_level_N en cada nodo para cada nivel N.
        Devuelve la lista de niveles detectados [0, 1, 2, ...].
        """
        result = self.graph.query(
            f"""
            CALL gds.leiden.write(
                '{self.gds_graph_name}',
                {{
                    writeProperty:                  'communityIds',
                    randomSeed:                     42,
                    gamma:                          1.0,
                    theta:                          0.01,
                    maxLevels:                      {self.max_levels},
                    tolerance:                      0.0001,
                    includeIntermediateCommunities: true
                }}
            )
            YIELD communityCount, modularity, ranLevels
            RETURN communityCount, modularity, ranLevels
            """
        )
        if not result:
            raise RuntimeError("Leiden no devolvió resultado.")

        row = result[0]
        ran_levels = row.get("ranLevels", 1)
        if self.verbose:
            print(f"   Modularidad: {row['modularity']:.4f}, niveles ejecutados: {ran_levels}")

        # GDS escribe communityIds come lista [id_livello0, id_livello1, ...]
        # Separiamo in proprietà distinte per facilità di query
        for level in range(ran_levels):
            self.graph.query(
                f"""
                MATCH (e:Entidad)
                WHERE e.communityIds IS NOT NULL
                SET e.communityId_level_{level} = e.communityIds[{level}]
                """
            )

        # Rimuovi la proprietà array intermedia
        self.graph.query("MATCH (e:Entidad) REMOVE e.communityIds")

        return list(range(ran_levels))

    def _eliminar_proyeccion_gds(self, silent: bool = False):
        try:
            self.graph.query(
                f"CALL gds.graph.drop('{self.gds_graph_name}', false) YIELD graphName"
            )
        except Exception:
            pass

    # ── Lectura de comunidades por nivel ─────────────────────────────────────

    def _leer_comunidades_nivel(self, level: int) -> List[Dict[str, Any]]:
        """
        Lee los nodos del nivel indicado desde Neo4j.
        Usa la propiedad communityId_level_{level} escrita por Leiden.
        """
        prop = f"communityId_level_{level}"
        result = self.graph.query(
            f"""
            MATCH (e:Entidad)
            WHERE e.{prop} IS NOT NULL
            WITH e.{prop} AS community_id,
                 collect({{
                     id:          e.id,
                     text:        e.text,
                     entity_type: e.entity_type,
                     descripcion: coalesce(e.descripcion_consolidada, e.descripcion, ""),
                     n_chunks:    size([(e)-[:MENTIONED_IN]->() | 1])
                 }}) AS miembros,
                 count(e) AS n
            ORDER BY n DESC
            RETURN community_id, miembros, n
            """
        )
        return [
            {"community_id": row["community_id"],
             "miembros":     row["miembros"],
             "n":            row["n"]}
            for row in result
        ]

    # ── Generación de summaries ───────────────────────────────────────────────

    def _generar_summary(self, comunidad: Dict, level: int) -> str:
        """Genera un summary temático adaptado al nivel de granularidad."""
        miembros        = comunidad["miembros"]
        miembros_sorted = sorted(miembros, key=lambda m: -m.get("n_chunks", 0))
        miembros_prompt = miembros_sorted[: self.max_nodos_en_prompt]

        prompt   = self._build_community_prompt(miembros_prompt, len(miembros), level)
        response = self.client.chat.completions.create(
            model=_MODEL,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()

    def _build_community_prompt(
        self,
        miembros: List[Dict],
        total_miembros: int,
        level: int = 0,
    ) -> str:
        """
        Construye el prompt adaptado al nivel:
          - Nivel 0 (fino):   pide detalle específico del proceso
          - Nivel 1 (medio):  pide síntesis de varios procesos relacionados
          - Nivel 2+ (grueso): pide descripción del dominio temático macro
        """
        lineas = []
        for m in miembros:
            desc  = m.get("descripcion", "").strip()
            linea = f"- [{m['entity_type']}] \"{m['text']}\""
            if desc:
                linea += f": {desc}"
            lineas.append(linea)

        lista_miembros = "\n".join(lineas)
        nota_truncado  = (f"\n(Mostrando {len(miembros)} de {total_miembros} entidades)"
                          if total_miembros > len(miembros) else "")

        if level == 0:
            instruccion = ("Genera un resumen específico (máximo 3 frases) que describa "
                           "el proceso o concepto concreto que une estas entidades.")
        elif level == 1:
            instruccion = ("Genera una síntesis (máximo 4 frases) que describa el área "
                           "temática que agrupa estos procesos y conceptos relacionados.")
        else:
            instruccion = ("Genera una descripción macro (máximo 4 frases) del dominio "
                           "normativo o administrativo que engloba todos estos elementos.")

        return f"""Eres un asistente especializado en normativa universitaria española.
Se te presenta un clúster de entidades (nivel {level}) extraídas de los reglamentos
de la Universidad de La Laguna.

Entidades del clúster:{nota_truncado}
{lista_miembros}

{instruccion}
El resumen debe ser útil para responder consultas sobre normativa universitaria.
Responde SOLO con el resumen, en español, sin preámbulo."""

    # ── Guardado en Neo4j ─────────────────────────────────────────────────────

    def _guardar_community_node(
        self,
        community_id: int,
        level: int,
        comunidad: Dict,
        summary: str,
    ):
        """
        Crea un nodo :Community {level: N} con el summary,
        y lo conecta a sus miembros con :BELONGS_TO.
        El id del nodo incluye el nivel para evitar colisiones tra livelli.
        """
        node_id      = f"L{level}_C{community_id}"
        miembros_ids = [m["id"] for m in comunidad["miembros"]]

        self.graph.query(
            """
            MERGE (c:Community {node_id: $node_id})
            SET c.community_id = $community_id,
                c.level        = $level,
                c.summary      = $summary,
                c.n_miembros   = $n_miembros,
                c.created_at   = timestamp()
            """,
            {
                "node_id":      node_id,
                "community_id": community_id,
                "level":        level,
                "summary":      summary,
                "n_miembros":   len(miembros_ids),
            },
        )

        # Connette Entidad → Community (solo livello 0)
        # I livelli superiori si connettono tramite PARENT_OF
        if level == 0:
            batch_size = 50
            for i in range(0, len(miembros_ids), batch_size):
                batch = miembros_ids[i : i + batch_size]
                self.graph.query(
                    """
                    MATCH (c:Community {node_id: $node_id})
                    UNWIND $ids AS eid
                    MATCH (e:Entidad {id: eid})
                    MERGE (e)-[:BELONGS_TO]->(c)
                    """,
                    {"node_id": node_id, "ids": batch},
                )

    def _conectar_niveles(self, level_child: int, level_parent: int):
        """
        Crea archi :PARENT_OF tra comunità di livelli adiacenti.
        Una Community di livello N punta alla Community di livello N+1
        se la maggioranza dei suoi membri appartiene a quella comunità superiore.
        """
        prop_child  = f"communityId_level_{level_child}"
        prop_parent = f"communityId_level_{level_parent}"

        self.graph.query(
            f"""
            MATCH (e:Entidad)
            WHERE e.{prop_child} IS NOT NULL AND e.{prop_parent} IS NOT NULL
            WITH e.{prop_child} AS cid_child, e.{prop_parent} AS cid_parent,
                 count(e) AS n
            WITH cid_child, cid_parent, n
            ORDER BY cid_child, n DESC
            WITH cid_child, collect(cid_parent)[0] AS dominant_parent
            MATCH (child:Community  {{community_id: cid_child,  level: {level_child}}})
            MATCH (parent:Community {{community_id: dominant_parent, level: {level_parent}}})
            MERGE (child)-[:PARENT_OF]->(parent)
            """
        )
        if self.verbose:
            print(f"   🔗 Conectados niveles {level_child} → {level_parent}")


# ── Uso típico ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from src.graph_building.entity_resolver import EntityResolver
    from src.graph_building.entity_summarizer import EntitySummarizer
    from src.graph_building.graph_builder import GraphBuilder

    ruta_json = sys.argv[1] if len(sys.argv) > 1 else "ner_resultados.json"

    # Fase A: resolución de entidades
    resolver   = EntityResolver(fuzzy_threshold=0.93, verbose=True)
    entity_map = resolver.resolve(ruta_json)

    # Construcción del grafo
    builder = GraphBuilder(
        verbose=True,
        solo_relaciones_validas=True,
        confianza_minima="medium",
        entity_map=entity_map,
    )
    builder.construir_desde_json(ruta_json, limpiar=True)

    # Fase B: summarización de entidades
    summarizer = EntitySummarizer(graph=builder.graph, verbose=True)
    summarizer.summarize(entity_map)

    # Fase C: detección de comunidades
    detector = CommunityDetector(graph=builder.graph, verbose=True)
    detector.detect_and_summarize()

    builder.estadisticas()