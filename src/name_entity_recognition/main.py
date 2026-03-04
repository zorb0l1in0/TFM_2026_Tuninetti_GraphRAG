"""
main.py
-------
Punto de entrada del pipeline NER de dos pasos.

Uso:
  python -m ner.main
  # oppure, dalla cartella padre:
  python main.py
"""

import json
import pandas as pd
from dotenv import load_dotenv

from .cargador_chunks import CargadorChunksCSV
from .pipeline import PipelineNERDosPasos

load_dotenv()


def main():
    RUTA_ONTOLOGIA = "../../data/ontology.yaml"
    RUTA_CSV       = "../../data/processed/chunks/chunks_con_embeddings.csv"

    # ── Cargar chunks ─────────────────────────────────────────────────────────
    cargador = CargadorChunksCSV(RUTA_CSV)
    cargador.resumen()

    chunks = cargador.cargar_chunks(
        max_chunks=10,
        # filtro_documento="Reglamento",
        # filtro_tipo="normativa",
    )

    if not chunks:
        print("❌ No se cargaron chunks. Verifica el CSV.")
        return

    # ── Pipeline NER ──────────────────────────────────────────────────────────
    pipeline = PipelineNERDosPasos(RUTA_ONTOLOGIA)

    todos_resultados = []
    for i, chunk in enumerate(chunks, 1):
        print(f"\n>>> Chunk {i}/{len(chunks)} | {chunk['titulo']} [{chunk['chunk_id']}]")

        resultado = pipeline.ejecutar(chunk["texto"])
        resultado.update({
            "chunk_id": chunk["chunk_id"],
            "titulo":   chunk["titulo"],
            "tipo":     chunk["tipo"],
            "fuente":   chunk["fuente"],
        })

        todos_resultados.append(resultado)
        print(pipeline.formatear_resultado(resultado))

    # ── Guardar salida ────────────────────────────────────────────────────────
    with open("ner_resultados.json", "w", encoding="utf-8") as f:
        json.dump(todos_resultados, f, ensure_ascii=False, indent=2)
    print("📄 JSON completo guardado: ner_resultados.json")

    filas = [
        {
            "chunk_id":      r.get("chunk_id"),
            "titulo":        r.get("titulo"),
            "tipo":          r.get("tipo"),
            "fuente":        r.get("fuente"),
            "texto_entidad": e.get("text"),
            "tipo_entidad":  e.get("entity_type"),
            "confianza":     e.get("confidence"),
        }
        for r in todos_resultados
        for e in r.get("entidades", [])
    ]

    if filas:
        pd.DataFrame(filas).to_csv("ner_entidades.csv", index=False)
        print(f"📊 CSV de entidades guardado: ner_entidades.csv ({len(filas)} filas)")


if __name__ == "__main__":
    main()