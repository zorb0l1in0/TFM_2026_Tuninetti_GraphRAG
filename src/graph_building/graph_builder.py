"""
graph_builder.py
----------------
Construye un grafo Neo4j directamente desde ner_resultados.json,
usando el EntityMap producido por entity_resolver.py para crear
nodos canónicos en lugar de nodos por texto raw.

Cambios respecto a la versión anterior:
  - Acepta un EntityMap opcional en el constructor.
  - _merge_entidad: usa canonical_id + canonical_text del resolver.
  - _merge_relacion: resuelve sujeto/objeto vía canonical_id,
    no por búsqueda toLower en el grafo.

Estructura esperada de ner_resultados.json:
[
  {
    "chunk_id":   "chunk_0001",
    "titulo":     "Reglamento de matrícula",
    "tipo":       "normativa",
    "fuente":     "reglamento.pdf",
    "texto":      "...",
    "entidades":  [{"text": "...", "entity_type": "...", "confidence": "high", "descripcion": "..."}],
    "relaciones": [{"sujeto": "...", "relacion": "...", "objeto": "...", "valida": true, "motivo": ""}]
  },
  ...
]
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..common.clients import get_neo4j_graph
from .entity_resolver import EntityMap, EntityResolver



class GraphBuilder:
    """
    Carga ner_resultados.json y puebla Neo4j con:
      - Nodos :Chunk            (uno por elemento del JSON)
      - Nodos :Entidad          (uno por entidad canónica, label adicional = entity_type)
      - Arcos :MENTIONED_IN     (Entidad → Chunk)
      - Arcos :<RELACION>       (Entidad → Entidad, solo valida=True)

    Si se proporciona un EntityMap, los nodos se crean por canonical_id
    y todas las variantes textuales del mismo referente quedan fusionadas.
    """

    # Lista de textos genéricos a filtrar (si filtrar_genericos=True)
    _TEXTOS_GENERICOS = set()

    def __init__(
        self,
        verbose: bool = True,
        solo_relaciones_validas: bool = True,
        confianza_minima: str = "medium",
        filtrar_genericos: bool = False,
        min_longitud_texto: int = 5,
        entity_map: Optional[EntityMap] = None,  # ← NUEVO
    ):
        """
        Args:
            entity_map: EntityMap producido por EntityResolver.resolve().
                        Si es None, el builder funciona como antes (sin resolución).
        """
        self.verbose                 = verbose
        self.solo_relaciones_validas = solo_relaciones_validas
        self.confianza_minima        = confianza_minima
        self.filtrar_genericos       = filtrar_genericos
        self.min_longitud_texto      = min_longitud_texto
        self.entity_map              = entity_map  # ← NUEVO
        self._confianza_orden        = {"high": 3, "medium": 2, "low": 1}

        self.graph = get_neo4j_graph()


    # ── Punto de entrada ──────────────────────────────────────────────────────

    def construir_desde_json(
        self,
        ruta_json: str,
        limpiar: bool = True,
    ) -> Tuple[int, int, int]:
        """
        Carga ner_resultados.json y puebla el grafo.

        Args:
            ruta_json: ruta al JSON producido por la pipeline NER
            limpiar:   si True vacía el grafo antes de empezar

        Returns:
            (n_chunks, n_entidades, n_relaciones)
        """
        ruta = Path(ruta_json)
        if not ruta.exists():
            raise FileNotFoundError(f"JSON no encontrado: {ruta}")

        with open(ruta, encoding="utf-8") as f:
            resultados: List[Dict] = json.load(f)

        print(f"\n🕸️  Construyendo grafo desde {ruta.name} ({len(resultados)} chunks)...")

        if limpiar:
            self.graph.query("MATCH (n) DETACH DELETE n")
            if self.verbose:
                print("✅ Grafo limpiado")

        self._crear_constraints()

        total_entidades  = 0
        total_relaciones = 0

        for resultado in resultados:
            ent, rel = self._procesar_resultado(resultado)
            total_entidades  += ent
            total_relaciones += rel

        print(f"\n✅ Grafo construido:")
        print(f"   📦 Chunks     : {len(resultados)}")
        print(f"   🏷️  Entidades  : {total_entidades}")
        print(f"   🔗 Relaciones : {total_relaciones}")
        return (len(resultados), total_entidades, total_relaciones)

    # ── Procesamiento ─────────────────────────────────────────────────────────

    def _procesar_resultado(self, resultado: Dict) -> Tuple[int, int]:
        """Procesa un único elemento del JSON (= un chunk NER)."""
        chunk_id = resultado.get("chunk_id", "")
        titulo   = resultado.get("titulo",   "Sin título")
        tipo     = resultado.get("tipo",     "Desconocido")
        fuente   = resultado.get("fuente",   "")
        texto    = resultado.get("texto",    "")

        # 1. Crea nodo Chunk
        self._merge_chunk(chunk_id, titulo, tipo, fuente, texto)

        # 2. Filtra y deduplica entidades del chunk
        entidades = self._filtrar_y_deduplicar(resultado.get("entidades", []))

        # 3. Crea nodos Entidad + arcos :MENTIONED_IN
        for entidad in entidades:
            self._merge_entidad(entidad, chunk_id)

        # 4. Crea arcos de relación entre entidades
        relaciones = resultado.get("relaciones", [])
        if self.solo_relaciones_validas:
            relaciones = [r for r in relaciones if r.get("valida", False)]

        rel_create = 0
        for rel in relaciones:
            ok = self._merge_relacion(rel)
            if ok:
                rel_create += 1

        if self.verbose:
            print(f"   [{chunk_id}] {len(entidades)} entidades, {rel_create} relaciones")

        return (len(entidades), rel_create)

    # ── Cypher helpers ────────────────────────────────────────────────────────

    def _crear_constraints(self):
        queries = [
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
            "CREATE CONSTRAINT entidad_id IF NOT EXISTS FOR (e:Entidad) REQUIRE e.id IS UNIQUE",
        ]
        for q in queries:
            try:
                self.graph.query(q)
            except Exception:
                pass

    def _merge_chunk(self, chunk_id, titulo, tipo, fuente, texto):
        self.graph.query(
            """
            MERGE (c:Chunk {chunk_id: $chunk_id})
            SET c.titulo = $titulo,
                c.tipo   = $tipo,
                c.fuente = $fuente,
                c.texto  = $texto
            """,
            {"chunk_id": chunk_id, "titulo": titulo,
             "tipo": tipo, "fuente": fuente, "texto": texto},
        )

    def _merge_entidad(self, entidad: Dict, chunk_id: str):
        """
        Crea o actualiza el nodo Entidad y el arco :MENTIONED_IN hacia el Chunk.

        CON entity_map:
          - el id del nodo es canonical_id (hash SHA1, estable entre chunks)
          - el texto es canonical_text (variante más frecuente)
          - todas las variantes del mismo referente apuntan al mismo nodo

        SIN entity_map (fallback):
          - comportamiento original: id = entity_type::text_lower
        """
        text        = entidad.get("text", "").strip()
        entity_type = entidad.get("entity_type", "Desconocido")
        confidence  = entidad.get("confidence",  "low")
        descripcion = entidad.get("descripcion", "")

        if not text:
            return

        # ── Resolución canónica ───────────────────────────────────────────────
        if self.entity_map is not None:
            ce = self.entity_map.lookup(text)
            if ce is not None:
                entity_id      = ce.canonical_id
                canonical_text = ce.canonical_text
                entity_type    = ce.entity_type  # tipo del canónico, no del raw
            else:
                # Entidad no presente en el mapa (texto filtrado por min_length, etc.)
                entity_id      = f"{entity_type}::{text.lower()}"
                canonical_text = text
        else:
            # Fallback: comportamiento original sin resolver
            entity_id      = f"{entity_type}::{text.lower()}"
            canonical_text = text
        # ─────────────────────────────────────────────────────────────────────

        self.graph.query(
            """
            MERGE (e:Entidad {id: $id})
            ON CREATE SET e.text        = $text,
                          e.entity_type = $entity_type,
                          e.confidence  = $confidence,
                          e.descripcion = $descripcion
            ON MATCH  SET e.text        = $text,
                          e.entity_type = $entity_type
            WITH e
            MATCH (c:Chunk {chunk_id: $chunk_id})
            MERGE (e)-[:MENTIONED_IN]->(c)
            """,
            {
                "id":          entity_id,
                "text":        canonical_text,
                "entity_type": entity_type,
                "confidence":  confidence,
                "descripcion": descripcion,
                "chunk_id":    chunk_id,
            },
        )

        # Label dinámico por tipo (requiere APOC)
        try:
            self.graph.query(
                "MATCH (e:Entidad {id: $id}) CALL apoc.create.addLabels(e, [$label]) YIELD node RETURN node",
                {"id": entity_id, "label": entity_type},
            )
        except Exception:
            pass





    def _merge_relacion(self, rel: Dict) -> bool:
        sujeto = rel.get("sujeto", "").strip()
        objeto = rel.get("objeto", "").strip()
        relacion = rel.get("relacion", "").strip().upper().replace(" ", "_")

        if not sujeto or not objeto or not relacion:
            return False

        # Elimina auto-loop por texto idéntico
        if sujeto.lower() == objeto.lower():
            if self.verbose:
                print(f"     ⚠️ Auto-loop eliminado: '{sujeto}' --[{relacion}]--> '{objeto}'")
            return False

        # ── Resolución canónica ───────────────────────────────────────────────
        if self.entity_map is not None:
            src_id = self.entity_map.canonical_id(sujeto)
            tgt_id = self.entity_map.canonical_id(objeto)

            if not src_id or not tgt_id:
                if self.verbose:
                    print(f"     ⚠️ Relación '{relacion}': extremo no resuelto "
                          f"('{sujeto}' → '{objeto}')")
                return False

            # Elimina auto-loop post-canonicalizzazione
            if src_id == tgt_id:
                if self.verbose:
                    print(f"     ⚠️ Auto-loop canónico eliminado: '{sujeto}' --[{relacion}]--> '{objeto}'")
                return False

            try:
                self.graph.query(
                    f"""
                    MATCH (s:Entidad {{id: $src_id}})
                    MATCH (o:Entidad {{id: $tgt_id}})
                    MERGE (s)-[r:`{relacion}`]->(o)
                    RETURN r
                    """,
                    {"src_id": src_id, "tgt_id": tgt_id},
                )
                return True
            except Exception as e:
                if self.verbose:
                    print(f"     ⚠️ Relación '{relacion}': {e}")
                return False
        # ─────────────────────────────────────────────────────────────────────

        # Fallback: comportamiento original sin resolver
        try:
            self.graph.query(
                f"""
                MATCH (s:Entidad) WHERE toLower(s.text) = toLower($sujeto)
                MATCH (o:Entidad) WHERE toLower(o.text) = toLower($objeto)
                MERGE (s)-[r:`{relacion}`]->(o)
                RETURN r
                """,
                {"sujeto": sujeto, "objeto": objeto},
            )
            return True
        except Exception as e:
            if self.verbose:
                print(f"     ⚠️ Relación '{relacion}': {e}")
            return False




    # ── Filtro y deduplicación ────────────────────────────────────────────────

    def _filtrar_y_deduplicar(self, entidades: List[Dict]) -> List[Dict]:
        """
        Por cada chunk:
        1. Descarta entidades bajo el umbral de confianza
        2. Descarta textos genéricos (si filtrar_genericos=True)
        3. Descarta textos demasiado cortos
        4. Deduplica por (text_normalizado, entity_type):
           en caso de duplicados conserva la confianza más alta,
           y en empate concatena las descripciones
        """
        resultado: Dict[str, Dict] = {}

        for e in entidades:
            text       = e.get("text", "").strip()
            etype      = e.get("entity_type", "")
            confidence = e.get("confidence", "low")

            if not self._confianza_ok(confidence):
                continue
            if len(text) < self.min_longitud_texto:
                continue
            if self.filtrar_genericos and text.lower() in self._TEXTOS_GENERICOS:
                continue

            key = f"{etype}::{text.lower()}"

            if key not in resultado:
                resultado[key] = dict(e)
            else:
                existing = resultado[key]
                if self._confianza_orden.get(confidence, 1) > \
                   self._confianza_orden.get(existing.get("confidence", "low"), 1):
                    resultado[key] = dict(e)
                elif e.get("descripcion") and e["descripcion"] != existing.get("descripcion"):
                    existing["descripcion"] = (existing.get("descripcion", "") +
                                               " / " + e["descripcion"])

        return list(resultado.values())

    def _confianza_ok(self, confidence: str) -> bool:
        minima = self._confianza_orden.get(self.confianza_minima, 1)
        actual = self._confianza_orden.get(confidence, 1)
        return actual >= minima

    # ── Utilidades ────────────────────────────────────────────────────────────

    def consultar(self, query: str, params: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Ejecuta una query Cypher libre."""
        return self.graph.query(query, params or {})

    def estadisticas(self) -> Dict[str, int]:
        """Devuelve estadísticas básicas del grafo."""
        stats = {}
        for label, key in [("Chunk", "chunks"), ("Entidad", "entidades")]:
            res = self.graph.query(f"MATCH (n:{label}) RETURN count(n) as n")
            stats[key] = res[0]["n"] if res else 0

        res = self.graph.query("MATCH ()-[r]->() RETURN count(r) as n")
        stats["relaciones"] = res[0]["n"] if res else 0

        print(f"\n📊 Estadísticas del grafo:")
        for k, v in stats.items():
            print(f"   • {k}: {v}")
        return stats


# ── Uso típico ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    ruta_json = sys.argv[1] if len(sys.argv) > 1 else "ner_resultados.json"

    # 1. Construye el EntityMap (Fase A del pipeline GraphRAG)
    resolver   = EntityResolver(embedding_threshold=0.92, verbose=True)
    entity_map = resolver.resolve(ruta_json)
    entity_map.print_stats()

    # 2. Construye el grafo usando los canónicos
    builder = GraphBuilder(
        verbose=True,
        solo_relaciones_validas=True,
        confianza_minima="medium",
        entity_map=entity_map,  # ← pasa el mapa aquí
    )
    builder.construir_desde_json(ruta_json, limpiar=True)
    builder.estadisticas()