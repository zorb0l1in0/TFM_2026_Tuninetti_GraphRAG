"""
relation_canonicalizer.py
--------------------------
Implementa la fase "Canonicalize" del framework EDC
(Extract, Define, Canonicalize — Zhang & Soh, arXiv:2404.03868).

Problema che risolve:
  Il pipeline NER estrae relazioni semanticamente corrette ma non presenti
  nell'ontologia (es. EXPEDIR, CERTIFICAR, FIGURAR con rango sbagliato).
  Il VerificadorRestricciones le scarta, perdendo informazione utile.

Soluzione EDC:
  1. Raccoglie tutte le relazioni NON valide dal ner_resultados.json
  2. Per ciascuna, calcola l'embedding del suo nome
  3. Confronta con gli embedding dei nomi delle relazioni ontologiche
  4. Se la similarità coseno supera la soglia → mappa alla relazione più vicina
  5. Se sotto soglia → scarta (relazione genuinamente sbagliata)
  6. Zona grigia (tra soglia_bassa e soglia_alta) → log CSV per revisione umana

Uso tipico (dopo NER, prima di GraphBuilder):
    canonicalizer = RelationCanonicalizer(ruta_ontologia="ontology.yaml")
    n_mappate, n_scartate = canonicalizer.canonicalizar(
        ruta_ner_json="ner_resultados.json",
        ruta_output="ner_resultados.json",   # sovrascrive in-place
    )

Uso da CLI:
    python -m src.name_entity_recognition.relation_canonicalizer \\
        --ner data/ner/ner_resultados.json \\
        --ontologia data/ner/ontologia/ontology.yaml
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

from src.common.clients import get_embeddings


# ── Costanti ──────────────────────────────────────────────────────────────────

# Soglia principale: sopra → mappata, sotto → scartata
_SOGLIA_DEFAULT       = 0.75

# Soglia "zona grigia": tra _SOGLIA_GRIGIA e _SOGLIA_DEFAULT → log per revisione
_SOGLIA_GRIGIA_DEFAULT = 0.60


# ── RelationCanonicalizer ─────────────────────────────────────────────────────

class RelationCanonicalizer:
    """
    Canonicalizza le relazioni non-ontologiche estratte dal pipeline NER
    tramite similarità coseno sugli embedding dei nomi di relazione.

    Implementa il passo "Canonicalize" del framework EDC con:
      - soglia principale per accettare automaticamente la mappatura
      - zona grigia per revisione umana (export CSV)
      - scarto sotto soglia minima
    """

    def __init__(
        self,
        ruta_ontologia:   str,
        soglia:           float = _SOGLIA_DEFAULT,
        soglia_grigia:    float = _SOGLIA_GRIGIA_DEFAULT,
        verbose:          bool  = True,
    ):
        """
        Args:
            ruta_ontologia:  path al file ontology.yaml
            soglia:          similarità minima per mappatura automatica (default 0.75)
            soglia_grigia:   soglia inferiore della zona grigia (default 0.60)
                             relazioni con sim tra soglia_grigia e soglia → log CSV
            verbose:         stampa log di progresso
        """
        self.soglia        = soglia
        self.soglia_grigia = soglia_grigia
        self.verbose       = verbose

        # Carica relazioni ontologiche
        self.relazioni_onto: List[str] = self._cargar_relaciones_ontologia(ruta_ontologia)

        # Calcola embedding delle relazioni ontologiche (una volta sola)
        if self.verbose:
            print(f"📐 RelationCanonicalizer — embedding {len(self.relazioni_onto)} relazioni ontologiche...")
        vecs = get_embeddings(self.relazioni_onto)
        mat  = np.array(vecs, dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        self.mat_onto_norm = mat / (norms + 1e-9)  # (N_onto, dim)

        if self.verbose:
            print(f"   Relazioni ontologiche: {self.relazioni_onto}")

    # ── Punto di ingresso ─────────────────────────────────────────────────────

    def canonicalizar(
        self,
        ruta_ner_json: str,
        ruta_output:   Optional[str] = None,
        ruta_log_csv:  Optional[str] = None,
    ) -> Tuple[int, int, int]:
        """
        Processa il file ner_resultados.json e canonicalizza le relazioni
        non valide.

        Args:
            ruta_ner_json:  path al JSON prodotto dal pipeline NER
            ruta_output:    path di output (default: sovrascrive l'input)
            ruta_log_csv:   path per il CSV della zona grigia
                            (default: stesso dir del JSON, suffisso _zona_grigia.csv)

        Returns:
            (n_mappate, n_zona_grigia, n_scartate)
        """
        ruta_json = Path(ruta_ner_json)
        if not ruta_json.exists():
            raise FileNotFoundError(f"JSON non trovato: {ruta_json}")

        ruta_out = Path(ruta_output) if ruta_output else ruta_json
        ruta_csv = Path(ruta_log_csv) if ruta_log_csv else (
            ruta_json.parent / (ruta_json.stem + "_zona_grigia.csv")
        )

        with open(ruta_json, encoding="utf-8") as f:
            chunks: List[Dict] = json.load(f)

        # Raccoglie tutte le relazioni non valide uniche
        rel_non_valide = self._raccogliere_relazioni_non_valide(chunks)

        if not rel_non_valide:
            if self.verbose:
                print("✅ Nessuna relazione non valida trovata — nulla da canonicalizzare.")
            return (0, 0, 0)

        if self.verbose:
            print(f"\n🔗 Canonicalizzazione EDC — {len(rel_non_valide)} tipi di relazione non validi")

        # Calcola embedding delle relazioni non valide (batch unico)
        nomi_non_validi = list(rel_non_valide)
        vecs_nv = get_embeddings(nomi_non_validi)
        mat_nv  = np.array(vecs_nv, dtype=np.float32)
        norms_nv = np.linalg.norm(mat_nv, axis=1, keepdims=True)
        mat_nv_norm = mat_nv / (norms_nv + 1e-9)  # (N_nv, dim)

        # Matrice di similarità coseno: (N_nv, N_onto)
        sim_matrix = mat_nv_norm @ self.mat_onto_norm.T

        # Costruisce mappa: relazione_non_valida → (relazione_canonica, sim, zona)
        mappa:      Dict[str, str]   = {}   # mappate automaticamente
        zona_grigia: List[Dict]      = []   # da revisionare
        n_mappate   = 0
        n_grigia    = 0
        n_scartate  = 0

        for i, nome_nv in enumerate(nomi_non_validi):
            sims       = sim_matrix[i]       # (N_onto,)
            idx_best   = int(np.argmax(sims))
            sim_best   = float(sims[idx_best])
            onto_best  = self.relazioni_onto[idx_best]

            if sim_best >= self.soglia:
                # Mappatura automatica
                mappa[nome_nv] = onto_best
                n_mappate += 1
                if self.verbose:
                    print(f"   ✅ {nome_nv} → {onto_best}  (sim={sim_best:.3f})")

            elif sim_best >= self.soglia_grigia:
                # Zona grigia: log per revisione umana
                zona_grigia.append({
                    "relacion_extraida":   nome_nv,
                    "canonico_proposto":   onto_best,
                    "similitudine":        round(sim_best, 4),
                    "seconda_scelta":      self._segunda_scelta(sims, idx_best),
                    "decisione":           "",   # campo da compilare manualmente
                })
                n_grigia += 1
                if self.verbose:
                    print(f"   ⚠️  {nome_nv} → {onto_best}  (sim={sim_best:.3f}) — ZONA GRIGIA")

            else:
                # Scartata: troppo distante da qualunque relazione ontologica
                n_scartate += 1
                if self.verbose:
                    print(f"   ✗  {nome_nv}  (sim_max={sim_best:.3f}) — SCARTATA")

        # Applica la mappa al JSON
        chunks_modificati = self._applicar_mappa(chunks, mappa)

        # Salva JSON modificato
        with open(ruta_out, "w", encoding="utf-8") as f:
            json.dump(chunks_modificati, f, ensure_ascii=False, indent=2)

        # Salva CSV zona grigia (se ci sono relazioni da revisionare)
        if zona_grigia:
            self._salvar_csv_zona_grigia(zona_grigia, ruta_csv)

        # Riepilogo
        print(f"\n✅ Canonicalizzazione EDC completata:")
        print(f"   • Mappate automaticamente : {n_mappate}")
        print(f"   • Zona grigia (revisione) : {n_grigia}  → {ruta_csv.name}")
        print(f"   • Scartate                : {n_scartate}")
        print(f"   • JSON aggiornato         : {ruta_out.name}")

        return (n_mappate, n_grigia, n_scartate)

    # ── Helpers interni ───────────────────────────────────────────────────────

    def _cargar_relaciones_ontologia(self, ruta_yaml: str) -> List[str]:
        """Carica i nomi delle relazioni dall'ontologia YAML."""
        with open(ruta_yaml, encoding="utf-8") as f:
            onto = yaml.safe_load(f)
        return list(onto.get("relations", {}).keys())

    def _raccogliere_relazioni_non_valide(self, chunks: List[Dict]) -> set:
        """Raccoglie solo le relazioni il cui NOME non esiste nell'ontologia.
        Le relazioni con nome valido ma domain/range errato vengono lasciate
        per revisione manuale nel tool."""
        nomi_onto = set(self.relazioni_onto)
        non_valide = set()
        for chunk in chunks:
            for rel in chunk.get("relaciones", []):
                if not rel.get("valida", True):
                    nombre = rel.get("relacion", "").strip()
                    if nombre and nombre != "NONE" and nombre not in nomi_onto:
                        non_valide.add(nombre)
        return non_valide

    def _applicar_mappa(self, chunks: List[Dict], mappa: Dict[str, str]) -> List[Dict]:
        """
        Applica la mappa di canonicalizzazione al JSON.
        Le relazioni mappate diventano valide con la relazione canonica.
        Le relazioni non mappate (zona grigia o scartate) rimangono non valide.
        """
        if not mappa:
            return chunks

        for chunk in chunks:
            relaciones_nuove = []
            for rel in chunk.get("relaciones", []):
                if not rel.get("valida", True):
                    nombre = rel.get("relacion", "")
                    if nombre in mappa:
                        # Sostituisce il tipo e marca come valida
                        rel_nuova = dict(rel)
                        rel_nuova["relacion"]          = mappa[nombre]
                        rel_nuova["valida"]            = True
                        rel_nuova["relacion_original"] = nombre   # conserva il tipo originale
                        rel_nuova["motivo"]            = ""
                        relaciones_nuove.append(rel_nuova)
                    else:
                        relaciones_nuove.append(rel)
                else:
                    relaciones_nuove.append(rel)
            chunk["relaciones"] = relaciones_nuove

        return chunks

    def _segunda_scelta(self, sims: np.ndarray, idx_best: int) -> str:
        """Restituisce il nome della seconda relazione più simile."""
        sims_copia = sims.copy()
        sims_copia[idx_best] = -1.0
        idx2 = int(np.argmax(sims_copia))
        return f"{self.relazioni_onto[idx2]} ({sims_copia[idx2]:.3f})"

    def _salvar_csv_zona_grigia(self, zona_grigia: List[Dict], ruta_csv: Path):
        """Salva il CSV della zona grigia per revisione umana."""
        ruta_csv.parent.mkdir(parents=True, exist_ok=True)
        campos = ["relacion_extraida", "canonico_proposto", "similitudine",
                  "seconda_scelta", "decisione"]
        with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=campos)
            writer.writeheader()
            writer.writerows(zona_grigia)
        print(f"   📋 CSV zona grigia salvato: {ruta_csv}")
        print(f"      → Compila la colonna 'decisione' con il tipo canonico scelto")
        print(f"         o lascia vuoto per scartare la relazione.")


# ── Integrazione con il pipeline NER ─────────────────────────────────────────

def canonicalizar_post_ner(
    ruta_ner_json:  str,
    ruta_ontologia: str,
    soglia:         float = _SOGLIA_DEFAULT,
    soglia_grigia:  float = _SOGLIA_GRIGIA_DEFAULT,
    verbose:        bool  = True,
) -> Tuple[int, int, int]:
    """
    Funzione di convenienza per integrare la canonicalizzazione
    direttamente nel master_pipeline.py.

    Esempio in paso_grafo():
        from src.name_entity_recognition.relation_canonicalizer import canonicalizar_post_ner
        canonicalizar_post_ner(
            ruta_ner_json  = str(RUTA_RESULTADOS),
            ruta_ontologia = str(RUTA_ONTOLOGIA),
        )
    """
    canonicalizer = RelationCanonicalizer(
        ruta_ontologia = ruta_ontologia,
        soglia         = soglia,
        soglia_grigia  = soglia_grigia,
        verbose        = verbose,
    )
    return canonicalizer.canonicalizar(ruta_ner_json)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Canonicalizzazione EDC delle relazioni non-ontologiche"
    )
    parser.add_argument("--ner",       required=True, help="Path a ner_resultados.json")
    parser.add_argument("--ontologia", required=True, help="Path a ontology.yaml")
    parser.add_argument("--soglia",    type=float, default=_SOGLIA_DEFAULT,
                        help=f"Soglia mappatura automatica (default={_SOGLIA_DEFAULT})")
    parser.add_argument("--soglia-grigia", type=float, default=_SOGLIA_GRIGIA_DEFAULT,
                        help=f"Soglia zona grigia (default={_SOGLIA_GRIGIA_DEFAULT})")
    parser.add_argument("--output",    default=None, help="Path output JSON (default: sovrascrive input)")
    parser.add_argument("--log-csv",   default=None, help="Path CSV zona grigia")
    parser.add_argument("--quiet",     action="store_true", help="Disabilita log verbose")
    args = parser.parse_args()

    canonicalizer = RelationCanonicalizer(
        ruta_ontologia = args.ontologia,
        soglia         = args.soglia,
        soglia_grigia  = args.soglia_grigia,
        verbose        = not args.quiet,
    )
    canonicalizer.canonicalizar(
        ruta_ner_json = args.ner,
        ruta_output   = args.output,
        ruta_log_csv  = args.log_csv,
    )