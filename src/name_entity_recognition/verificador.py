"""
verificador.py
--------------
Verifica post-hoc che le relazioni rispettino i vincoli domain/range
dell'ontologia (Chepurova et al., TextGraphs 2024).
"""

from .ontologia import CargadorOntologia
from .schemas import RelacionItem


class VerificadorRestricciones:
    """
    Verifica post-hoc que las relaciones respeten las restricciones domain/range
    de la ontología.
    """

    def __init__(self, ontologia: CargadorOntologia, mapa_entidades: dict[str, str]):
        self.restricciones  = ontologia.restricciones_relaciones()
        self.mapa_entidades = mapa_entidades

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
                motivo = f"Relación '{rel.relacion}' no definida en la ontología"
            else:
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