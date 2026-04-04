"""
microsoft_graphrag.py
---------------------
Implementazione delle due modalità di retrieval del paper Microsoft GraphRAG
(Edge et al., 2024 — arxiv.org/abs/2404.16130).

  LOCAL SEARCH
    Risponde a domande specifiche su entità nominate.
    Flusso:
      1. Embedding query → top-k entità più simili (anchor)
      2. Estrae il contesto locale: chunk dove appare l'entità,
         relazioni uscenti/entranti, community report dell'entità
      3. Assembla un context window e chiama il LLM per la risposta

  GLOBAL SEARCH
    Risponde a domande tematiche/comparative sull'intero corpus.
    Flusso:
      1. Recupera i community report a tutti i livelli gerarchici
      2. Per ogni community, chiama il LLM per generare una risposta parziale
         (map step)
      3. Aggrega le risposte parziali con un secondo LLM (reduce step)

Nota sull'implementazione:
  - I community report vengono generati dal CommunityDetector che già hai.
    Questo script li legge da Neo4j dove sono già salvati.
  - La gerarchia di community è quella prodotta da Leiden via GDS.
  - Non viene usato un vector store esterno: tutto è in Neo4j + memory.

Metriche: solo qualitative (risposta del LLM) — non P/R/F1 perché
  local/global search producono risposte generate, non chunk ranked.
  Per confronto quantitativo usare local_retriever.py.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.common.clients import get_embeddings, get_neo4j_graph, get_langchain_llm

# ── Rutas ─────────────────────────────────────────────────────────────────────

_ROOT       = Path(__file__).resolve().parent.parent.parent
RUTA_SALIDA = _ROOT / "data" / "retrieval" / "graphrag_microsoft.json"
RUTA_GT     = _ROOT / "data" / "retrieval" / "ground_truth.json"

# ── Costanti ──────────────────────────────────────────────────────────────────

TOP_K_ENTITA        = 5    # entità anchor per local search
MAX_CHUNK_PER_ENTITA = 3   # chunk per entità anchor
MAX_RELAZIONI       = 10   # relazioni da includere nel contesto locale
MAX_COMMUNITY_REPORT = 3   # report di community per entità (local)
MAP_BATCH_SIZE      = 5    # community per batch nel map step (global)

PREGUNTAS_LOCAL = [
    "¿Ante quién se interpone el recurso de alzada contra la resolución de cambio de turno?",
    "¿Cuáles son las causas justificadas que permiten solicitar el cambio de turno?",
    "¿Qué documentación debe aportar un estudiante que solicita cambio de turno por motivos laborales?",
    "¿Qué porcentaje mínimo de créditos debe superar un estudiante para mantenerse en el Programa de Doble Grado?",
    "¿Qué órgano aprueba el calendario anual para la tramitación de nuevos Programas Académicos de Doble Grado?",
    "¿Cuál es la fórmula para calcular la nota de admisión del alumnado de Bachillerato español?",
    "¿Qué requisito de reconocimiento de créditos debe cumplirse para acceder por traslado de expediente?",
    "¿Qué norma habilita a la Secretaría General de la ULL para interpretar el reglamento de cambio de turno?",
    "¿Qué ocurre con la doble titulación si se extingue uno de los títulos de Grado que la compone?",
    "¿Qué universidades firman la resolución conjunta sobre requisitos académicos de admisión 2025-2026?",
]

PREGUNTAS_GLOBAL = [
    "¿Cuáles son los principales procedimientos administrativos regulados en estos reglamentos?",
    "¿Qué tipos de entidades institucionales aparecen con más frecuencia y qué roles tienen?",
    "¿Cuáles son los derechos y obligaciones principales del estudiantado según la normativa?",
]


# ═════════════════════════════════════════════════════════════════════════════
# INDICE ENTITÀ — embedding precalcolato all'init
# ═════════════════════════════════════════════════════════════════════════════

class IndiceEntita:
    """
    Calcola e mantiene in memoria gli embedding di tutte le entità del grafo.
    Usato sia da LocalSearch che come base per GlobalSearch.
    """

    def __init__(self, graph, verbose: bool = True):
        self.graph   = graph
        self.verbose = verbose

        if verbose:
            print("   [IndiceEntita] Caricamento entità dal grafo...")

        rows = graph.query(
            "MATCH (e:Entidad) RETURN e.id AS id, e.text AS text, "
            "e.entity_type AS tipo LIMIT 5000"
        )
        self.ids   = [r["id"]   for r in rows]
        self.testi = [r["text"] for r in rows]
        self.tipi  = [r["tipo"] for r in rows]

        if verbose:
            print(f"   [IndiceEntita] Calcolo embedding {len(self.testi)} entità...")

        batch_size = 50
        vecs = []
        for i in range(0, len(self.testi), batch_size):
            vecs.extend(get_embeddings(self.testi[i:i+batch_size]))

        mat   = np.array(vecs, dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        self.mat_norm = mat / (norms + 1e-9)

        if verbose:
            print(f"   [IndiceEntita] Pronto. {len(self.ids)} entità indicizzate.")

    def cerca(self, query: str, top_k: int = TOP_K_ENTITA) -> List[Dict]:
        """Restituisce le top_k entità più simili alla query."""
        vec   = np.array(get_embeddings([query])[0], dtype=np.float32)
        vec_n = vec / (np.linalg.norm(vec) + 1e-9)
        sims  = self.mat_norm @ vec_n
        idxs  = np.argsort(sims)[::-1][:top_k]
        return [
            {"id": self.ids[i], "text": self.testi[i],
             "tipo": self.tipi[i], "sim": round(float(sims[i]), 3)}
            for i in idxs
        ]


# ═════════════════════════════════════════════════════════════════════════════
# LOCAL SEARCH
# ═════════════════════════════════════════════════════════════════════════════

class LocalSearch:
    """
    Implementa la Local Search di Microsoft GraphRAG.

    Per ogni query:
      1. Trova le entità anchor per similarità embedding
      2. Per ogni anchor recupera:
         - chunk dove è menzionata (testo sorgente)
         - relazioni uscenti e entranti (struttura del grafo)
         - community report dell'entità (sintesi di alto livello)
      3. Assembla un context window strutturato
      4. Chiama il LLM con il contesto per generare la risposta

    Il context window segue il formato del paper:
      [Entities] → [Relationships] → [Sources] → [Community Reports]
    """

    def __init__(
        self,
        graph,
        indice:         IndiceEntita,
        llm=None,
        top_k_entita:   int  = TOP_K_ENTITA,
        max_chunk:      int  = MAX_CHUNK_PER_ENTITA,
        max_relazioni:  int  = MAX_RELAZIONI,
        max_community:  int  = MAX_COMMUNITY_REPORT,
        verbose:        bool = True,
    ):
        self.graph         = graph
        self.indice        = indice
        self.llm           = llm or get_langchain_llm()
        self.top_k_entita  = top_k_entita
        self.max_chunk     = max_chunk
        self.max_relazioni = max_relazioni
        self.max_community = max_community
        self.verbose       = verbose

    def cerca(self, query: str) -> Dict:
        t0 = time.time()

        # 1. Anchor
        anchor = self.indice.cerca(query, self.top_k_entita)
        anchor_ids = [a["id"] for a in anchor]

        if self.verbose:
            for a in anchor:
                print(f"      anchor: [{a['tipo']}] '{a['text'][:50]}' (sim={a['sim']})")

        # 2. Contesto locale
        chunks     = self._recupera_chunks(anchor_ids)
        relazioni  = self._recupera_relazioni(anchor_ids)
        community  = self._recupera_community(anchor_ids)

        # 3. Assembla context window
        context = self._build_context(anchor, chunks, relazioni, community)

        # 4. Chiama LLM
        risposta = self._genera_risposta(query, context)

        return {
            "query":      query,
            "anchor":     anchor,
            "n_chunks":   len(chunks),
            "n_relazioni": len(relazioni),
            "n_community": len(community),
            "context_len": len(context),
            "risposta":   risposta,
            "tiempo_s":   round(time.time() - t0, 2),
        }

    def _recupera_chunks(self, ids: List[str]) -> List[Dict]:
        rows = self.graph.query(
            """
            MATCH (e:Entidad)-[:MENTIONED_IN]->(c:Chunk)
            WHERE e.id IN $ids
            RETURN DISTINCT c.chunk_id AS chunk_id,
                   c.titulo AS titulo,
                   c.texto  AS texto
            LIMIT $lim
            """,
            {"ids": ids, "lim": self.max_chunk * len(ids)},
        )
        return rows

    def _recupera_relazioni(self, ids: List[str]) -> List[Dict]:
        """
        Recupera relazioni uscenti e entranti delle entità anchor.
        Include il tipo di relazione e le entità coinvolte.
        """
        rows = self.graph.query(
            """
            MATCH (e:Entidad)-[r]->(o:Entidad)
            WHERE e.id IN $ids
              AND type(r) <> 'MENTIONED_IN' AND type(r) <> 'BELONGS_TO'
            RETURN e.text AS sujeto, type(r) AS relacion,
                   o.text AS objeto, r.descripcion AS descripcion
            LIMIT $lim
            UNION
            MATCH (s:Entidad)-[r]->(e:Entidad)
            WHERE e.id IN $ids
              AND type(r) <> 'MENTIONED_IN' AND type(r) <> 'BELONGS_TO'
            RETURN s.text AS sujeto, type(r) AS relacion,
                   e.text AS objeto, r.descripcion AS descripcion
            LIMIT $lim
            """,
            {"ids": ids, "lim": self.max_relazioni},
        )
        return rows

    def _recupera_community(self, ids: List[str]) -> List[Dict]:
        """
        Recupera i community report delle community a cui appartengono
        le entità anchor. Usa la proprietà summary già scritta da CommunityDetector.
        """
        rows = self.graph.query(
            """
            MATCH (e:Entidad)-[:BELONGS_TO]->(c:Community)
            WHERE e.id IN $ids AND c.summary IS NOT NULL
            RETURN DISTINCT c.community_id AS community_id,
                   c.summary AS summary,
                   c.size    AS size
            ORDER BY c.size DESC
            LIMIT $lim
            """,
            {"ids": ids, "lim": self.max_community},
        )
        return rows

    def _build_context(
        self,
        anchor:    List[Dict],
        chunks:    List[Dict],
        relazioni: List[Dict],
        community: List[Dict],
    ) -> str:
        parts = []

        # Entità
        parts.append("## ENTIDADES RELEVANTES")
        for a in anchor:
            parts.append(f"- [{a['tipo']}] {a['text']} (similitud={a['sim']})")

        # Relazioni
        if relazioni:
            parts.append("\n## RELACIONES")
            for r in relazioni:
                desc = f" — {r['descripcion']}" if r.get("descripcion") else ""
                parts.append(f"- {r['sujeto']} --[{r['relacion']}]--> {r['objeto']}{desc}")

        # Testi sorgente
        if chunks:
            parts.append("\n## FRAGMENTOS DE TEXTO FUENTE")
            for c in chunks[:self.max_chunk]:
                texto = (c.get("texto") or "")[:500]
                parts.append(f"[{c.get('titulo','')[:60]}]\n{texto}")

        # Community report
        if community:
            parts.append("\n## RESÚMENES DE COMUNIDAD")
            for cm in community:
                parts.append(f"[Comunidad {cm['community_id']} — {cm['size']} nodos]\n{cm['summary']}")

        return "\n".join(parts)

    def _genera_risposta(self, query: str, context: str) -> str:
        prompt = f"""Eres un asistente especializado en normativa universitaria española.
Usa ÚNICAMENTE la información del contexto siguiente para responder la pregunta.
Si la información no está en el contexto, dilo explícitamente.

{context}

---
PREGUNTA: {query}

Responde de forma concisa y precisa, citando las fuentes cuando sea posible."""

        try:
            res = self.llm.invoke(prompt)
            return res.content.strip()
        except Exception as e:
            return f"[Error LLM: {e}]"


# ═════════════════════════════════════════════════════════════════════════════
# GLOBAL SEARCH
# ═════════════════════════════════════════════════════════════════════════════

class GlobalSearch:
    """
    Implementa la Global Search di Microsoft GraphRAG.

    Per ogni query:
      1. Recupera tutti i community report dal grafo (tutti i livelli)
      2. MAP: per ogni batch di community, chiama il LLM per generare
         una risposta parziale con un punteggio di utilità [0-100]
      3. REDUCE: filtra le risposte con utilità > soglia, le aggrega
         con un secondo LLM per produrre la risposta finale

    Differenza con Local Search: non usa entità anchor, lavora su tutto
    il corpus tramite i community report — adatta per domande globali
    su temi, pattern, confronti tra documenti.
    """

    def __init__(
        self,
        graph,
        llm=None,
        batch_size:     int   = MAP_BATCH_SIZE,
        soglia_utilita: int   = 20,    # ignora risposte parziali sotto questa soglia
        verbose:        bool  = True,
    ):
        self.graph          = graph
        self.llm            = llm or get_langchain_llm()
        self.batch_size     = batch_size
        self.soglia_utilita = soglia_utilita
        self.verbose        = verbose

        # Carica tutti i community report al init
        if verbose:
            print("   [GlobalSearch] Caricamento community report...")
        self.community_reports = self._carica_reports()
        if verbose:
            print(f"   [GlobalSearch] {len(self.community_reports)} community report caricati.")

    def _carica_reports(self) -> List[Dict]:
        rows = self.graph.query(
            """
            MATCH (c:Community)
            WHERE c.summary IS NOT NULL
            RETURN c.community_id AS id, c.summary AS summary,
                   c.size AS size, c.level AS level
            ORDER BY c.level ASC, c.size DESC
            """
        )
        return rows

    def cerca(self, query: str) -> Dict:
        t0 = time.time()

        if not self.community_reports:
            return {
                "query":    query,
                "risposta": "Nessun community report disponibile. Esegui prima il CommunityDetector.",
                "n_community": 0,
                "tiempo_s": round(time.time() - t0, 2),
            }

        # MAP step: genera risposte parziali per batch di community
        risposte_parziali = []
        batches = [
            self.community_reports[i:i+self.batch_size]
            for i in range(0, len(self.community_reports), self.batch_size)
        ]

        if self.verbose:
            print(f"      MAP: {len(batches)} batch di {self.batch_size} community...")

        for i, batch in enumerate(batches):
            risposta_parziale = self._map_step(query, batch, i+1, len(batches))
            if risposta_parziale and risposta_parziale["utilita"] >= self.soglia_utilita:
                risposte_parziali.append(risposta_parziale)

        if self.verbose:
            print(f"      REDUCE: {len(risposte_parziali)} risposte utili su {len(batches)} batch")

        # REDUCE step: aggrega le risposte parziali
        risposta_finale = self._reduce_step(query, risposte_parziali)

        return {
            "query":        query,
            "n_community":  len(self.community_reports),
            "n_batch":      len(batches),
            "n_utili":      len(risposte_parziali),
            "risposta":     risposta_finale,
            "tiempo_s":     round(time.time() - t0, 2),
        }

    def _map_step(self, query: str, batch: List[Dict], i: int, tot: int) -> Optional[Dict]:
        """
        Per un batch di community report, chiede al LLM di generare una
        risposta parziale e un punteggio di utilità [0-100].
        """
        context = "\n\n".join(
            f"[Comunidad {r['id']} — nivel {r.get('level','?')}, {r['size']} nodos]\n{r['summary']}"
            for r in batch
        )

        prompt = f"""Eres un asistente de análisis documental.
A continuación tienes resúmenes de comunidades de un grafo de conocimiento.
Usando SOLO esta información, genera una respuesta parcial a la pregunta.
Si la información no es relevante, indica utilidad=0.

Formato de respuesta (JSON estricto):
{{
  "respuesta": "tu respuesta parcial aquí, o vacío si no es relevante",
  "utilidad": <número entre 0 y 100 que indica cuán útil es esta información para la pregunta>
}}

COMUNIDADES:
{context}

PREGUNTA: {query}

Responde SOLO con el JSON, sin texto adicional."""

        try:
            res  = self.llm.invoke(prompt)
            text = res.content.strip()
            # Pulisce possibili markdown fences
            text = text.replace("```json", "").replace("```", "").strip()
            data = json.loads(text)
            if self.verbose:
                print(f"        batch {i}/{tot} → utilità={data.get('utilita', 0)}")
            return {"respuesta": data.get("respuesta", ""), "utilita": int(data.get("utilita", 0))}
        except Exception as e:
            if self.verbose:
                print(f"        batch {i}/{tot} → errore: {e}")
            return None

    def _reduce_step(self, query: str, risposte_parziali: List[Dict]) -> str:
        """
        Aggrega le risposte parziali filtrate in una risposta finale coerente.
        """
        if not risposte_parziali:
            return "No se encontró información relevante en los resúmenes de comunidad disponibles."

        testi = "\n\n---\n\n".join(
            f"[Utilidad: {r['utilita']}/100]\n{r['respuesta']}"
            for r in sorted(risposte_parziali, key=lambda x: -x["utilita"])
        )

        prompt = f"""Eres un asistente especializado en normativa universitaria española.
A continuación tienes varias respuestas parciales generadas a partir de distintas
partes del corpus documental. Sintetízalas en una respuesta final coherente y completa.
Elimina redundancias y contradicciones. Sé conciso pero completo.

RESPUESTAS PARCIALES:
{testi}

PREGUNTA ORIGINAL: {query}

Responde directamente, sin preámbulo."""

        try:
            res = self.llm.invoke(prompt)
            return res.content.strip()
        except Exception as e:
            return f"[Error en reduce step: {e}]"


# ═════════════════════════════════════════════════════════════════════════════
# RUNNER
# ═════════════════════════════════════════════════════════════════════════════

def esegui_local(preguntas: List[str], local: LocalSearch) -> List[Dict]:
    risultati = []
    for i, q in enumerate(preguntas, 1):
        print(f"\n{'='*65}")
        print(f"[LOCAL {i}/{len(preguntas)}] {q[:70]}")
        print("="*65)
        r = local.cerca(q)
        print(f"\n  Anchor: {[a['text'][:30] for a in r['anchor']]}")
        print(f"  Contesto: {r['n_chunks']} chunk, {r['n_relazioni']} relazioni, {r['n_community']} community")
        print(f"  Tempo: {r['tiempo_s']}s")
        print(f"\n  RISPOSTA:\n  {r['risposta'][:300]}{'...' if len(r['risposta'])>300 else ''}")
        risultati.append(r)
    return risultati


def esegui_global(preguntas: List[str], global_: GlobalSearch) -> List[Dict]:
    risultati = []
    for i, q in enumerate(preguntas, 1):
        print(f"\n{'='*65}")
        print(f"[GLOBAL {i}/{len(preguntas)}] {q[:70]}")
        print("="*65)
        r = global_.cerca(q)
        print(f"\n  Community: {r['n_community']} totali, {r['n_utili']} utili, {r['n_batch']} batch")
        print(f"  Tempo: {r['tiempo_s']}s")
        print(f"\n  RISPOSTA:\n  {r['risposta'][:400]}{'...' if len(r['risposta'])>400 else ''}")
        risultati.append(r)
    return risultati


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*65)
    print("🔬 MICROSOFT GRAPHRAG — Local Search + Global Search")
    print("="*65)

    graph = get_neo4j_graph()
    llm   = get_langchain_llm()

    print("\n⚙️  Init indice entità...")
    indice = IndiceEntita(graph, verbose=True)

    print("\n⚙️  Init Local Search...")
    local = LocalSearch(graph=graph, indice=indice, llm=llm, verbose=True)

    print("\n⚙️  Init Global Search...")
    global_ = GlobalSearch(graph=graph, llm=llm, verbose=True)

    # LOCAL SEARCH
    print("\n" + "="*65)
    print("📍 LOCAL SEARCH")
    print("="*65)
    risultati_local = esegui_local(PREGUNTAS_LOCAL, local)

    # GLOBAL SEARCH
    print("\n" + "="*65)
    print("🌐 GLOBAL SEARCH")
    print("="*65)
    risultati_global = esegui_global(PREGUNTAS_GLOBAL, global_)

    # Salva
    output = {
        "local_search":  risultati_local,
        "global_search": risultati_global,
    }
    RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    with open(RUTA_SALIDA, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Salvato: {RUTA_SALIDA}")