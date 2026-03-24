"""
pipeline.py
-----------
Pipeline NER de dos pasos con verificación ontológica.

  Paso 1  — Detección de spans   (sin tipo)
  Paso 2  — Clasificación        (guiada por ontología)
  Paso 2b — Verificación         (comprobación de restricciones)

La ontología SOLO puede proporcionarse mediante:
  A) YAML existente → PipelineNERDosPasos(ruta_ontologia="ontology.yaml")
"""

import sys
import json
import threading
import hashlib
from pathlib import Path
from typing import Optional
from datetime import datetime

from langchain_core.callbacks import BaseCallbackHandler
from .ontologia import CargadorOntologia
from .verificador import VerificadorRestricciones
from .schemas import RelacionItem
from .prompts import construir_cadena_paso1, construir_cadena_paso2
from dotenv import load_dotenv
from ..common.clients import get_langchain_llm_paso1, get_langchain_llm_paso2

load_dotenv()

# Directorio para logs de respuestas LLM
_RUTA_LOG_LLM = Path("../data/ner/llm_responses")
_RUTA_LOG_LLM.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------
#   LOGGER DE RESPUESTAS LLM
# ----------------------------------------------------------
class LLMResponseLogger(BaseCallbackHandler):
    """Guarda la respuesta del LLM en un archivo .txt"""

    def __init__(self, prefix: str, texto: str):
        self.prefix = prefix
        self.texto = texto

    def _filename(self) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        short_hash = hashlib.sha1(self.texto.encode("utf-8")).hexdigest()[:10]
        return _RUTA_LOG_LLM / f"{self.prefix}_{short_hash}_{ts}.txt"

    def on_llm_end(self, response, **kwargs):
        text = None
        try:
            gen0 = response.generations[0][0]
            text = getattr(gen0, "text", None) or getattr(gen0, "message", None)
            if hasattr(text, "content"):
                text = text.content
        except Exception:
            pass
        if text is None:
            text = str(response)
        try:
            with open(self._filename(), "w", encoding="utf-8") as f:
                f.write(text)
        except Exception as e:
            print(f"[LOG LLM] Error escribiendo log: {e}", flush=True)


# ----------------------------------------------------------
#   PIPELINE NER DE DOS PASOS
# ----------------------------------------------------------
class PipelineNERDosPasos:

    def __init__(
        self,
        ruta_ontologia: Optional[str] = None,
        temperatura: float = 0.0,
        **kwargs
    ):
        if not ruta_ontologia:
            raise ValueError("Debes proporcionar ruta_ontologia (ontology.yaml).")

        self.ontologia = CargadorOntologia(ruta_ontologia)
        self.cadena_paso1 = construir_cadena_paso1(get_langchain_llm_paso1(temperatura))
        self.cadena_paso2 = construir_cadena_paso2(get_langchain_llm_paso2(temperatura), self.ontologia)

    # ----------------------------------------------------------
    #   HEARTBEAT
    # ----------------------------------------------------------
    def _llamar_con_heartbeat(self, cadena, inputs: dict, callbacks: list, label: str = "LLM"):
        resultado = [None]
        errore = [None]
        stop = threading.Event()

        def heartbeat():
            t = 0
            while not stop.is_set():
                stop.wait(15)
                if not stop.is_set():
                    t += 15
                    print(f"  ⏳ [{label}] Esperando respuesta LLM... ({t}s)", flush=True)

        threading.Thread(target=heartbeat, daemon=True).start()

        try:
            resultado[0] = cadena.invoke(inputs, config={"callbacks": callbacks})
        except Exception as e:
            errore[0] = e
        finally:
            stop.set()

        if errore[0]:
            raise errore[0]
        return resultado[0]

    # ----------------------------------------------------------
    #   PASO 1 — DETECCIÓN DE SPANS
    # ----------------------------------------------------------
    def ejecutar_paso1(self, texto: str) -> list[dict]:
        print("  [Paso 1] Llamando al LLM...", flush=True)
        cb = LLMResponseLogger(prefix="paso1", texto=texto)
        resultado = self._llamar_con_heartbeat(self.cadena_paso1, {"text": texto}, [cb], label="Paso1")
        print("  [Paso 1] Respuesta recibida.", flush=True)

        if not isinstance(resultado, dict):
            print("  ⚠ Paso 1: la respuesta no es un dict válido.", flush=True)
            return []

        spans = resultado.get("spans", [])
        print(f"  [Paso 1] {len(spans)} spans candidatos encontrados.", flush=True)
        return spans

    # ----------------------------------------------------------
    #   PASO 2 — CLASIFICACIÓN + VERIFICACIÓN
    # ----------------------------------------------------------
    def ejecutar_paso2(self, texto: str, spans: list[dict]):
        print("  [Paso 2] Llamando al LLM...", flush=True)
        spans_json = json.dumps(spans, ensure_ascii=False, indent=2)
        cb = LLMResponseLogger(prefix="paso2", texto=texto)
        resultado = self._llamar_con_heartbeat(
            self.cadena_paso2,
            {"text": texto, "spans_json": spans_json},
            [cb],
            label="Paso2"
        )
        print("  [Paso 2] Respuesta recibida.", flush=True)

        entidades_raw = resultado.get("entidades", [])
        relaciones_raw = resultado.get("relaciones", [])

        tipos_validos = self.ontologia.tipos_entidad_validos()
        entidades = [
            e for e in entidades_raw
            if e.get("entity_type") in tipos_validos and e.get("entity_type") not in ("NONE", None)
        ]

        mapa_entidades = {e["text"]: e["entity_type"] for e in entidades}
        items_relacion = [
            RelacionItem(
                sujeto=r.get("sujeto", ""),
                relacion=r.get("relacion", ""),
                objeto=r.get("objeto", ""),
            )
            for r in relaciones_raw
        ]

        verificador = VerificadorRestricciones(self.ontologia, mapa_entidades)
        relaciones = verificador.verificar(items_relacion)

        validas = sum(1 for r in relaciones if r.get("valida"))
        print(f"  [Paso 2] {len(entidades)} entidades, {validas}/{len(relaciones)} relaciones válidas.", flush=True)
        return entidades, relaciones

    # ----------------------------------------------------------
    #   EJECUCION COMPLETA POR CHUNK
    # ----------------------------------------------------------
    def ejecutar(self, texto: str) -> dict:
        spans = self.ejecutar_paso1(texto)

        if not spans:
            print("  ⚠ Sin spans — chunk omitido.", flush=True)
            return {"texto": texto, "entidades": [], "relaciones": [], "spans_paso1": []}

        entidades, relaciones = self.ejecutar_paso2(texto, spans)

        print(f"\n  📌 SPANS ({len(spans)}):", flush=True)
        for s in spans:
            print(f'     • "{s.get("text", "")}"', flush=True)

        print(f"\n  🏷️  ENTIDADES ({len(entidades)}):", flush=True)
        if entidades:
            for e in entidades:
                icono = {"high": "✓", "medium": "~", "low": "?"}.get(e.get("confidence", "?"), "?")
                print(f'     {icono} [{e.get("entity_type")}] "{e.get("text")}"', flush=True)
        else:
            print("     (ninguna entidad válida)", flush=True)

        print(f"\n  🔗 RELACIONES ({len(relaciones)}):", flush=True)
        if relaciones:
            for r in relaciones:
                ok = "✓" if r.get("valida") else "✗"
                linea = f'     {ok} {r.get("sujeto")} --[{r.get("relacion")}]--> {r.get("objeto")}'
                if not r.get("valida") and r.get("motivo"):
                    linea += f'\n        ⚠ {r["motivo"]}'
                print(linea, flush=True)
        else:
            print("     (ninguna relación extraída)", flush=True)

        return {
            "texto": texto,
            "spans_paso1": spans,
            "entidades": entidades,
            "relaciones": relaciones,
        }