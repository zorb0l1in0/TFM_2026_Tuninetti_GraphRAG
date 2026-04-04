"""
local_retriever.py
------------------
Confronta due strategie di retrieval locale:

  A) EMBEDDING PURO
     Similarità coseno query → chunk direttamente dal CSV.

  B) NER + HOP GRAFO
     1. NER sulla query con pipeline LLM (paso1+paso2)
     2. Lookup nel grafo per testo esatto/fuzzy
     3. Traversal + scoring per decay hop

Metriche: Precision@K, Recall@K, F1@K su ground_truth.json
"""

from __future__ import annotations

import ast
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.common.clients import get_embeddings, get_neo4j_graph
from src.name_entity_recognition.pipeline import PipelineNERDosPasos
from src.graph_building.entity_resolver import EntityMap, EntityResolver

# ── Rutas ─────────────────────────────────────────────────────────────────────

_ROOT          = Path(__file__).resolve().parent.parent.parent
RUTA_CSV       = _ROOT / "data" / "processed" / "chunks" / "chunks_con_embeddings.csv"
RUTA_SALIDA    = _ROOT / "data" / "retrieval" / "local_retriever.json"
RUTA_ONTOLOGIA = _ROOT / "data" / "ner" / "ontologia" / "ontology.yaml"
RUTA_NER_JSON  = _ROOT / "data" / "ner" / "ner_resultados.json"
RUTA_ACRONIMOS = _ROOT / "data" / "acronimos.yaml"
RUTA_GT        = _ROOT / "data" / "retrieval" / "ground_truth.json"

# ── Costanti ──────────────────────────────────────────────────────────────────

TOP_K_CHUNKS = 5
SCORE_DECAY  = 0.5
MAX_HOPS     = 2

PREGUNTAS = [
    "¿Ante quién se interpone el recurso de alzada contra la resolución de cambio de turno?",
    "¿Cuáles son las causas justificadas que permiten solicitar el cambio de turno?",
    "¿Qué documentación debe aportar un estudiante que solicita cambio de turno por motivos laborales?",
    "¿Qué porcentaje mínimo de créditos debe superar un estudiante para mantenerse en el Programa de Doble Grado?",
    "¿Qué órgano aprueba el calendario anual para la tramitación de nuevos Programas Académicos de Doble Grado?",
    "¿Cuál es la fórmula para calcular la nota de admisión del alumnado de Bachillerato español?",
    "¿Qué requisito de reconocimiento de créditos debe cumplirse para acceder por traslado de expediente según la resolución conjunta?",
    "¿Qué norma habilita a la Secretaría General de la ULL para interpretar y resolver cuestiones sobre el reglamento de cambio de turno?",
    "¿Qué ocurre con la doble titulación si se extingue uno de los títulos de Grado que la compone?",
    "¿Qué universidades firman la resolución conjunta sobre requisitos académicos de admisión para el curso 2025-2026?"
]


# ═════════════════════════════════════════════════════════════════════════════
# DATACLASS
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class ChunkRecuperado:
    chunk_id: str
    titulo:   str
    score:    float = 1.0
    hop:      int   = 0


# ═════════════════════════════════════════════════════════════════════════════
# VALUTATORE
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class Metricas:
    precision:  float = 0.0
    recall:     float = 0.0
    f1:         float = 0.0
    tp:         List[str] = field(default_factory=list)
    relevantes: List[str] = field(default_factory=list)


class Evaluador:
    def __init__(self, ruta: Path = RUTA_GT):
        if not ruta.exists():
            raise FileNotFoundError(f"Ground truth non trovata: {ruta}")
        with open(ruta, encoding="utf-8") as f:
            dati = json.load(f)
        self._gt: Dict[str, List[str]] = {
            d["pregunta"]: d["chunks_relevantes"] for d in dati
        }

    def evaluar(self, pregunta: str, recuperados: List[str]) -> Metricas:
        relevantes = self._gt.get(pregunta, [])
        if not relevantes:
            return Metricas(relevantes=[], tp=[])
        tp = set(recuperados) & set(relevantes)
        k  = len(recuperados)
        p  = len(tp) / k if k else 0.0
        r  = len(tp) / len(relevantes) if relevantes else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return Metricas(
            precision=round(p, 3), recall=round(r, 3), f1=round(f1, 3),
            tp=sorted(tp), relevantes=relevantes,
        )

    def has_gt(self, pregunta: str) -> bool:
        return pregunta in self._gt


# ═════════════════════════════════════════════════════════════════════════════
# A — EMBEDDING PURO
# ═════════════════════════════════════════════════════════════════════════════

class RetrieverEmbedding:
    def __init__(self, ruta_csv: Path = RUTA_CSV, top_k: int = TOP_K_CHUNKS, verbose: bool = False):
        self.top_k = top_k
        df = pd.read_csv(ruta_csv)

        def parsear(v):
            return np.array(ast.literal_eval(v), dtype=np.float32) if isinstance(v, str) else np.array(v, dtype=np.float32)

        df["_emb"] = df["embedding"].apply(parsear)
        self.df    = df
        mat        = np.stack(df["_emb"].values)
        norms      = np.linalg.norm(mat, axis=1, keepdims=True)
        self.mat   = mat / (norms + 1e-9)
        if verbose:
            print(f"   [Embedding] {len(df)} chunks, dim={mat.shape[1]}")

    def recuperar(self, consulta: str) -> List[ChunkRecuperado]:
        vec   = np.array(get_embeddings([consulta])[0], dtype=np.float32)
        vec_n = vec / (np.linalg.norm(vec) + 1e-9)
        scores  = self.mat @ vec_n
        indices = np.argsort(scores)[::-1][:self.top_k]
        return [
            ChunkRecuperado(
                chunk_id=str(self.df.iloc[i].get("id_chunk", self.df.iloc[i].get("chunk_id", f"idx_{i}"))),
                titulo=str(self.df.iloc[i].get("titulo", "")),
                score=round(float(scores[i]), 4),
                hop=0,
            )
            for i in indices
        ]


# ═════════════════════════════════════════════════════════════════════════════
# B — NER + HOP GRAFO
# ═════════════════════════════════════════════════════════════════════════════

class RetrieverNERGrafo:
    def __init__(
        self,
        graph=None,
        entity_map: Optional[EntityMap] = None,
        top_k: int   = TOP_K_CHUNKS,
        max_hops: int = MAX_HOPS,
        ruta_ontologia: Path = RUTA_ONTOLOGIA,
        ruta_ner_json:  Path = RUTA_NER_JSON,
        ruta_acronimos: Path = RUTA_ACRONIMOS,
        verbose: bool = False,
    ):
        self.graph    = graph or get_neo4j_graph()
        self.top_k    = top_k
        self.max_hops = max_hops
        self.verbose  = verbose

        self.pipeline = PipelineNERDosPasos(
            ruta_ontologia=str(ruta_ontologia),
            temperatura=0.0,
        )
        if entity_map is None:
            resolver   = EntityResolver(usar_embedding_merge=False, ruta_acronimos=ruta_acronimos, verbose=False)
            entity_map = resolver.resolve(str(ruta_ner_json))
        self.entity_map = entity_map

    def recuperar(self, consulta: str) -> Tuple[List[ChunkRecuperado], List[str]]:
        entidades = self._ner(consulta)
        if not entidades:
            return [], []
        nodos = self._lookup(entidades)
        if not nodos:
            return [], entidades
        chunks_raw = self._traversal([n["id"] for n in nodos])
        return self._rankear(chunks_raw), entidades

    def _ner(self, consulta: str) -> List[str]:
        try:
            res = self.pipeline.ejecutar(consulta)
        except Exception as e:
            if self.verbose:
                print(f"      ⚠ NER error: {e}")
            return []

        textos = []

        # Entidades (como antes)
        for ent in res.get("entidades", []):
            testo = ent.get("text", "").strip()
            if not testo:
                continue
            canonical = self.entity_map.lookup(testo)
            testo_final = canonical.canonical_text if canonical else testo
            if testo_final not in textos:
                textos.append(testo_final)

        # Relaciones: sujeto + objeto SIN filtrar por valida
        for rel in res.get("relaciones", []):
            for campo in ("sujeto", "objeto"):
                testo = rel.get(campo, "").strip()
                if not testo:
                    continue
                canonical = self.entity_map.lookup(testo)
                testo_final = canonical.canonical_text if canonical else testo
                if testo_final not in textos:
                    textos.append(testo_final)

        return textos

    def _lookup(self, entidades: List[str]) -> List[Dict]:
        trovati   = []
        ids_visti = set()
        for testo in entidades:
            rows = self.graph.query(
                "MATCH (e:Entidad) WHERE toLower(e.text) = toLower($t) "
                "RETURN e.id AS id, e.text AS text LIMIT 5",
                {"t": testo},
            )
            if not rows:
                rows = self.graph.query(
                    "MATCH (e:Entidad) "
                    "WHERE toLower(e.text) CONTAINS toLower($t) "
                    "   OR toLower($t) CONTAINS toLower(e.text) "
                    "RETURN e.id AS id, e.text AS text "
                    "ORDER BY size(e.text) DESC LIMIT 3",
                    {"t": testo},
                )
            if self.verbose:
                if rows:
                    print(f"      🔍 '{testo}' → {[r['id'] for r in rows]}")
                else:
                    print(f"      ❌ '{testo}' → nessun nodo trovato nel grafo")
            for r in rows:
                if r["id"] not in ids_visti:
                    ids_visti.add(r["id"])
                    trovati.append(r)
        return trovati

    def _traversal(self, ids: List[str]) -> List[Dict]:
        risultati = []
        risultati.extend(self.graph.query(
            """
            MATCH (e:Entidad)-[:MENTIONED_IN]->(c:Chunk)
            WHERE e.id IN $ids
            RETURN c.chunk_id AS chunk_id, c.titulo AS titulo,
                   0 AS hop, 1.0 AS decay
            """,
            {"ids": ids},
        ))
        for hop in range(1, self.max_hops + 1):
            decay = SCORE_DECAY ** hop
            try:
                risultati.extend(self.graph.query(
                    f"""
                    MATCH path = (seed:Entidad)-[*1..{hop}]-(vicino:Entidad)
                    WHERE seed.id IN $ids
                      AND ALL(r IN relationships(path)
                              WHERE type(r) <> 'MENTIONED_IN' AND type(r) <> 'BELONGS_TO')
                    WITH DISTINCT vicino
                    MATCH (vicino)-[:MENTIONED_IN]->(c:Chunk)
                    RETURN c.chunk_id AS chunk_id, c.titulo AS titulo,
                           {hop} AS hop, {decay:.4f} AS decay
                    """,
                    {"ids": ids},
                ))
            except Exception as e:
                if self.verbose:
                    print(f"      ⚠ traversal hop={hop}: {e}")
        return risultati

    def _rankear(self, chunks_raw: List[Dict]) -> List[ChunkRecuperado]:
        agg: Dict[str, Dict] = {}
        for r in chunks_raw:
            cid = r["chunk_id"]
            if cid not in agg:
                agg[cid] = {"chunk_id": cid, "titulo": r.get("titulo", ""),
                             "score": 0.0, "hop": r.get("hop", 0)}
            agg[cid]["score"] += float(r.get("decay", 1.0))
            agg[cid]["hop"]    = min(agg[cid]["hop"], r.get("hop", 0))
        ordinati = sorted(agg.values(), key=lambda x: (-x["score"], x["hop"]))
        return [
            ChunkRecuperado(chunk_id=r["chunk_id"], titulo=r["titulo"],
                            score=round(r["score"], 4), hop=r["hop"])
            for r in ordinati[:self.top_k]
        ]


# ═════════════════════════════════════════════════════════════════════════════
# CONFRONTO
# ═════════════════════════════════════════════════════════════════════════════

def confrontar(
    preguntas: List[str],
    ret_emb:   RetrieverEmbedding,
    ret_ner:   RetrieverNERGrafo,
    evaluador: Optional[Evaluador] = None,
) -> List[Dict]:
    risultati = []

    for i, q in enumerate(preguntas, 1):
        print(f"\n{'='*65}")
        print(f"[{i}/{len(preguntas)}] {q[:70]}")
        print("="*65)

        # A — Embedding puro
        print("  📐 A) Embedding puro...")
        t0 = time.time()
        chunks_a = ret_emb.recuperar(q)
        t_a = round(time.time() - t0, 2)
        ids_a = [c.chunk_id for c in chunks_a]

        # B — NER + grafo
        print("  🏷️  B) NER + grafo...")
        t0 = time.time()
        chunks_b, entita_b = ret_ner.recuperar(q)
        t_b = round(time.time() - t0, 2)
        ids_b = [c.chunk_id for c in chunks_b]

        m_a = evaluador.evaluar(q, ids_a) if evaluador and evaluador.has_gt(q) else None
        m_b = evaluador.evaluar(q, ids_b) if evaluador and evaluador.has_gt(q) else None

        r = {
            "pregunta": q,
            "A_embedding": {
                "chunks":   [{"chunk_id": c.chunk_id, "titulo": c.titulo, "score": c.score} for c in chunks_a],
                "n":        len(chunks_a),
                "tiempo_s": t_a,
                "metricas": _m2dict(m_a),
            },
            "B_ner_grafo": {
                "chunks":    [{"chunk_id": c.chunk_id, "titulo": c.titulo, "score": c.score, "hop": c.hop} for c in chunks_b],
                "n":         len(chunks_b),
                "entidades": entita_b,
                "tiempo_s":  t_b,
                "metricas":  _m2dict(m_b),
            },
            "overlap": sorted(set(ids_a) & set(ids_b)),
        }
        risultati.append(r)
        _stampa(r)

    return risultati


def _m2dict(m: Optional[Metricas]) -> Optional[Dict]:
    if m is None:
        return None
    return {"precision": m.precision, "recall": m.recall, "f1": m.f1, "tp": m.tp}


def _stampa(r: Dict):
    a = r["A_embedding"]
    b = r["B_ner_grafo"]
    print(f"\n  {'Retriever':<20} {'N':>3}  {'Tempo':>6}  {'P@K':>6}  {'R@K':>6}  {'F1':>6}")
    print(f"  {'-'*52}")
    for nome, d in [("A) Embedding puro", a), ("B) NER+Grafo", b)]:
        m  = d.get("metricas") or {}
        print(f"  {nome:<20} {d['n']:>3}  {d['tiempo_s']:>5.1f}s"
              f"  {str(m.get('precision','-')):>6}"
              f"  {str(m.get('recall','-')):>6}"
              f"  {str(m.get('f1','-')):>6}")
    print(f"  Overlap: {r['overlap']}")


def _riepilogo(risultati: List[Dict]):
    print(f"\n{'='*65}")
    print("📊 RIEPILOGO GLOBALE")
    print("="*65)
    for nome, key in [("A) Embedding puro", "A_embedding"), ("B) NER+Grafo", "B_ner_grafo")]:
        metriche = [r[key]["metricas"] for r in risultati if r[key].get("metricas")]
        if not metriche:
            continue
        p  = round(sum(m["precision"] for m in metriche) / len(metriche), 3)
        rv = round(sum(m["recall"]    for m in metriche) / len(metriche), 3)
        f1 = round(sum(m["f1"]        for m in metriche) / len(metriche), 3)
        t  = round(sum(r[key]["tiempo_s"] for r in risultati), 2)
        print(f"  {nome:<22}  P={p:.3f}  R={rv:.3f}  F1={f1:.3f}  t_tot={t}s")

    print(f"\n  {'#':<3} {'A-F1':>6}  {'B-F1':>6}  Pregunta")
    print(f"  {'-'*52}")
    for i, r in enumerate(risultati, 1):
        fa = (r["A_embedding"]["metricas"] or {}).get("f1", "-")
        fb = (r["B_ner_grafo"]["metricas"]  or {}).get("f1", "-")
        print(f"  {i:<3} {str(fa):>6}  {str(fb):>6}  {r['pregunta'][:45]}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*65)
    print("🔬 LOCAL RETRIEVER — Embedding puro vs NER+Grafo")
    print("="*65)

    graph = get_neo4j_graph()

    print("\n⚙️  Init A) Embedding puro...")
    ret_a = RetrieverEmbedding(top_k=TOP_K_CHUNKS, verbose=True)

    print("\n⚙️  Init B) NER + grafo...")
    resolver   = EntityResolver(usar_embedding_merge=False, ruta_acronimos=RUTA_ACRONIMOS, verbose=False)
    entity_map = resolver.resolve(str(RUTA_NER_JSON))
    ret_b = RetrieverNERGrafo(graph=graph, entity_map=entity_map,
                               top_k=TOP_K_CHUNKS, max_hops=MAX_HOPS, verbose=True)

    evaluador = None
    if RUTA_GT.exists():
        print(f"\n📏 Ground truth: {RUTA_GT}")
        evaluador = Evaluador(RUTA_GT)
    else:
        print(f"\n⚠️  Ground truth non trovata")

    risultati = confrontar(PREGUNTAS, ret_a, ret_b, evaluador)
    _riepilogo(risultati)

    RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    with open(RUTA_SALIDA, "w", encoding="utf-8") as f:
        json.dump(risultati, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Salvato: {RUTA_SALIDA}")