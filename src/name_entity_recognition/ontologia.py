"""
ontologia.py
------------
Carga el YAML de la ontología y lo serializa para los prompts.
"""

import yaml


class CargadorOntologia:
    """Carga el YAML de la ontología y lo serializa para los prompts."""

    def __init__(self, ruta_yaml: str):
        with open(ruta_yaml, "r", encoding="utf-8") as f:
            self.ontologia = yaml.safe_load(f)

    def descripcion_entidades(self) -> str:
        lineas = []
        for nombre, datos in self.ontologia.get("entities", {}).items():
            patrones = ", ".join(datos.get("patterns", []))
            desc = datos.get("description", "")
            lineas.append(f"- **{nombre}**: {desc} (patrones: {patrones})")
        return "\n".join(lineas)

    def descripcion_relaciones(self) -> str:
        lineas = []
        for nombre, datos in self.ontologia.get("relations", {}).items():
            dominio = ", ".join(datos.get("domain", []))
            rango   = ", ".join(datos.get("range", []))
            desc    = datos.get("description", "")
            lineas.append(
                f"- **{nombre}**: {desc}  [dominio: {dominio}] → [rango: {rango}]"
            )
        return "\n".join(lineas)

    def texto_restricciones(self) -> str:
        return "\n".join(f"  - {c}" for c in self.ontologia.get("constraints", []))

    def tipos_entidad_validos(self) -> list[str]:
        return list(self.ontologia.get("entities", {}).keys())

    def restricciones_relaciones(self) -> dict:
        result = {}
        for nombre, datos in self.ontologia.get("relations", {}).items():
            entrada = {"domain": datos.get("domain", [])}
            if "range" in datos:
                entrada["range"] = datos["range"]
            result[nombre] = entrada
        return result