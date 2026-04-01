"""
verificador.py
--------------
Verifica post-hoc che le relazioni rispettino i vincoli domain/range
dell'ontologia (Chepurova et al., TextGraphs 2024).

Modifiche rispetto alla versione originale:
  - Aggiunta modalità SOFT (solo_existencia=True):
    valida solo che il nome della relazione esista nell'ontologia,
    senza controllare domain/range. Riduce i falsi negativi strutturali
    lasciando la correzione semantica alla fase EDC (relation_canonicalizer.py).
  - La modalità STRICT (default) mantiene il comportamento originale.
"""

from .ontologia import CargadorOntologia
from .schemas import RelacionItem


class VerificadorRestricciones:
    """
    Verifica post-hoc que las relaciones respeten las restricciones domain/range
    de la ontología.

    Args:
        ontologia:       CargadorOntologia con il YAML caricato
        mapa_entidades:  dict {texto_span: entity_type}
        solo_existencia: se True, valida solo che il nome della relazione
                         esista nell'ontologia (modalità SOFT).
                         se False (default), valida anche domain e range
                         (modalità STRICT).
    """

    def __init__(
        self,
        ontologia:       CargadorOntologia,
        mapa_entidades:  dict[str, str],
        solo_existencia: bool = False,
    ):
        self.restricciones   = ontologia.restricciones_relaciones()
        self.mapa_entidades  = mapa_entidades
        self.solo_existencia = solo_existencia

    def verificar(self, relaciones: list[RelacionItem]) -> list[dict]:
        """
        Devuelve lista de dicts con campos:
          sujeto, relacion, objeto, valida (bool), motivo (str)
        """
        resultados = []
        for rel in relaciones:
            restriccion = self.restricciones.get(rel.relacion)
            valida = False
            motivo = ""

            if restriccion is None:
                # Relazione non definita nell'ontologia
                motivo = f"Relación '{rel.relacion}' no definida en la ontología"

            elif self.solo_existencia:
                # Modalità SOFT: basta che il nome esista
                valida = True

            else:
                # Modalità STRICT: verifica anche domain e range
                tipo_sujeto = self.mapa_entidades.get(rel.sujeto, "")
                tipo_objeto = self.mapa_entidades.get(rel.objeto, "")

                domain_ok = tipo_sujeto in restriccion["domain"]
                rango_ok  = True
                if "range" in restriccion:
                    rango_ok = tipo_objeto in restriccion["range"]

                valida = domain_ok and rango_ok

                if not valida:
                    motivo = (
                        f"Esperado dominio={restriccion['domain']}, "
                        f"encontrado '{tipo_sujeto}'; "
                        f"esperado rango={restriccion.get('range', 'sin restricción')}, "
                        f"encontrado '{tipo_objeto}'"
                    )

            resultados.append({
                "sujeto":   rel.sujeto,
                "relacion": rel.relacion,
                "objeto":   rel.objeto,
                "valida":   valida,
                "motivo":   motivo if not valida else "",
            })
        return resultados