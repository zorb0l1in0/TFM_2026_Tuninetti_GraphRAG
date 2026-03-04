"""
graph_builder.py
----------------
Costruisce un grafo Neo4j direttamente da ner_resultados.json,
bypassando l'estrazione LLM (già fatta dalla pipeline NER).

Struttura attesa di ner_resultados.json:
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
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph

load_dotenv()


class GraphBuilder:
    """
    Carica ner_resultados.json e popola Neo4j con:
      - Nodi  :Chunk            (uno per elemento del JSON)
      - Nodi  :Entidad          (uno per entità unica, label aggiuntivo = entity_type)
      - Archi :MENTIONED_IN     (Entidad → Chunk)
      - Archi :<RELACION>       (Entidad → Entidad, solo valida=True)
    """

    # Testi troppo generici per essere nodi utili nel grafo
    _TEXTOS_GENERICOS = {
        "programa", "programas", "títulos", "título", "titulación",
        "titulaciones", "reglamento", "centro", "centros", "universidad",
        "propuestas", "propuesta", "estudios", "asignaturas", "asignatura",
        "curso", "cursos", "títulos", "créditos",
    }

    def __init__(
        self,
        verbose: bool = True,
        solo_relaciones_validas: bool = True,
        confianza_minima: str = "medium",  # "high" | "medium" | "low"
        filtrar_genericos: bool = True,
        min_longitud_texto: int = 5,
    ):
        """
        Args:
            verbose:                  Log dettagliato
            solo_relaciones_validas:  Se True ignora relazioni con valida=False
            confianza_minima:         Filtra entità sotto questa soglia
            filtrar_genericos:        Se True scarta entità con testo generico
            min_longitud_texto:       Lunghezza minima del testo entità
        """
        self.verbose                 = verbose
        self.solo_relaciones_validas = solo_relaciones_validas
        self.confianza_minima        = confianza_minima
        self.filtrar_genericos       = filtrar_genericos
        self.min_longitud_texto      = min_longitud_texto
        self._confianza_orden        = {"high": 3, "medium": 2, "low": 1}

        self.graph = self._conectar_neo4j()

    # ── Connessione ───────────────────────────────────────────────────────────

    def _conectar_neo4j(self) -> Neo4jGraph:
        uri      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
        username = os.getenv("NEO4J_USERNAME",  "neo4j")
        password = os.getenv("NEO4J_PASSWORD",  "")

        if not password:
            raise ValueError("NEO4J_PASSWORD no encontrada en .env")

        try:
            graph = Neo4jGraph(url=uri, username=username, password=password)
            if self.verbose:
                print(f"✅ Conectado a Neo4j: {uri}")
            return graph
        except Exception as e:
            raise ConnectionError(f"Error conectando a Neo4j: {e}") from e

    # ── Entry point ───────────────────────────────────────────────────────────

    def construir_desde_json(
        self,
        ruta_json: str,
        limpiar: bool = True,
    ) -> Tuple[int, int, int]:
        """
        Carica ner_resultados.json e popola il grafo.

        Args:
            ruta_json: percorso al file JSON prodotto dalla pipeline NER
            limpiar:   se True svuota il grafo prima di iniziare

        Returns:
            (n_chunks, n_entidades, n_relaciones)
        """
        ruta = Path(ruta_json)
        if not ruta.exists():
            raise FileNotFoundError(f"JSON non trovato: {ruta}")

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

        print(f"\n✅ Grafo costruito:")
        print(f"   📦 Chunks     : {len(resultados)}")
        print(f"   🏷️  Entidades  : {total_entidades}")
        print(f"   🔗 Relaciones : {total_relaciones}")
        return (len(resultados), total_entidades, total_relaciones)

    # ── Processing ────────────────────────────────────────────────────────────

    def _procesar_resultado(self, resultado: Dict) -> Tuple[int, int]:
        """Processa un singolo elemento del JSON (= un chunk NER)."""
        chunk_id = resultado.get("chunk_id", "")
        titulo   = resultado.get("titulo",   "Sin título")
        tipo     = resultado.get("tipo",     "Desconocido")
        fuente   = resultado.get("fuente",   "")
        texto    = resultado.get("texto",    "")

        # 1. Crea nodo Chunk
        self._merge_chunk(chunk_id, titulo, tipo, fuente, texto)

        # 2. Filtra e deduplica entità del chunk
        entidades = self._filtrar_y_deduplicar(resultado.get("entidades", []))

        # 3. Crea nodi Entidad + archi :MENTIONED_IN
        for entidad in entidades:
            self._merge_entidad(entidad, chunk_id)

        # 4. Crea archi relazione tra entità
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
        """Crea constraints e indici di base."""
        queries = [
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
            "CREATE CONSTRAINT entidad_id IF NOT EXISTS FOR (e:Entidad) REQUIRE e.id IS UNIQUE",
        ]
        for q in queries:
            try:
                self.graph.query(q)
            except Exception:
                pass  # Già esistenti

    def _merge_chunk(
        self,
        chunk_id: str,
        titulo: str,
        tipo: str,
        fuente: str,
        texto: str,
    ):
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
        Crea o aggiorna il nodo Entidad e l'arco :MENTIONED_IN verso il Chunk.
        Aggiunge un label dinamico uguale a entity_type (es. :Programa, :Normativa).
        """
        text        = entidad.get("text", "").strip()
        entity_type = entidad.get("entity_type", "Desconocido")
        confidence  = entidad.get("confidence",  "low")
        descripcion = entidad.get("descripcion", "")

        if not text:
            return

        # ID deterministico: type + text normalizzato
        entity_id = f"{entity_type}::{text.lower()}"

        # MERGE nodo base :Entidad
        self.graph.query(
            """
            MERGE (e:Entidad {id: $id})
            SET e.text        = $text,
                e.entity_type = $entity_type,
                e.confidence  = $confidence,
                e.descripcion = $descripcion
            WITH e
            MATCH (c:Chunk {chunk_id: $chunk_id})
            MERGE (e)-[:MENTIONED_IN]->(c)
            """,
            {
                "id":          entity_id,
                "text":        text,
                "entity_type": entity_type,
                "confidence":  confidence,
                "descripcion": descripcion,
                "chunk_id":    chunk_id,
            },
        )

        # Label dinamico per tipo (es. :Programa, :Normativa)
        # Neo4j non supporta label parametrici in MERGE, usiamo APOC se disponibile
        try:
            self.graph.query(
                "MATCH (e:Entidad {id: $id}) CALL apoc.create.addLabels(e, [$label]) YIELD node RETURN node",
                {"id": entity_id, "label": entity_type},
            )
        except Exception:
            pass  # APOC non disponibile, il label :Entidad è sufficiente

    def _merge_relacion(self, rel: Dict) -> bool:
        """
        Crea l'arco tra due nodi Entidad.
        Il tipo di relazione è il campo 'relacion' del JSON.
        Restituisce True se l'arco è stato creato.
        """
        sujeto   = rel.get("sujeto",   "").strip()
        objeto   = rel.get("objeto",   "").strip()
        relacion = rel.get("relacion", "").strip().upper().replace(" ", "_")

        if not sujeto or not objeto or not relacion:
            return False

        # Cerca i nodi per text (il match è case-insensitive sul text)
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

    # ── Filtro e deduplicazione ───────────────────────────────────────────────

    def _filtrar_y_deduplicar(self, entidades: List[Dict]) -> List[Dict]:
        """
        Per ogni chunk:
        1. Scarta entità sotto la soglia di confidenza
        2. Scarta testi generici (se filtrar_genericos=True)
        3. Scarta testi troppo corti
        4. Deduplica per (text_normalizzato, entity_type):
           in caso di duplicati tiene quello con confidenza più alta,
           a parità concatena le descrizioni
        """
        risultato: Dict[str, Dict] = {}  # key = "entity_type::text_lower"

        for e in entidades:
            text       = e.get("text", "").strip()
            etype      = e.get("entity_type", "")
            confidence = e.get("confidence", "low")

            # Filtro confidenza
            if not self._confianza_ok(confidence):
                continue

            # Filtro lunghezza
            if len(text) < self.min_longitud_texto:
                continue

            # Filtro testi generici
            if self.filtrar_genericos and text.lower() in self._TEXTOS_GENERICOS:
                continue

            key = f"{etype}::{text.lower()}"

            if key not in risultato:
                risultato[key] = dict(e)
            else:
                # Tieni la confidenza più alta
                existing = risultato[key]
                if self._confianza_orden.get(confidence, 1) > \
                   self._confianza_orden.get(existing.get("confidence", "low"), 1):
                    risultato[key] = dict(e)
                # Arricchisci la descrizione se diversa
                elif e.get("descripcion") and e["descripcion"] != existing.get("descripcion"):
                    existing["descripcion"] = existing.get("descripcion", "") + \
                                              " / " + e["descripcion"]

        return list(risultato.values())

    def _confianza_ok(self, confidence: str) -> bool:
        minima = self._confianza_orden.get(self.confianza_minima, 1)
        actual = self._confianza_orden.get(confidence, 1)
        return actual >= minima

    # ── Utilità ───────────────────────────────────────────────────────────────

    def consultar(self, query: str, params: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Esegue una query Cypher libera."""
        return self.graph.query(query, params or {})

    def estadisticas(self) -> Dict[str, int]:
        """Restituisce statistiche di base sul grafo."""
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
