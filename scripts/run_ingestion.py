#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de ingestión de datos - Versión ligera
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.ingestion.loader import CSVLoader


def main():
    """Función principal de ingestión"""

    print("\n🚀 INGESTIÓN DE DATOS")
    print("-" * 30)

    # 1. Cargar CSV
    NOMBRE_CSV = "chunks.csv"

    try:
        cargador = CSVLoader(NOMBRE_CSV)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return

    # 2. Mostrar resumen
    print(f"\n📊 RESUMEN:")
    print(f"   Total chunks: {len(cargador.df)}")

    # Tipos de documento
    if 'clase_documento' in cargador.df.columns:
        print(f"\n📑 Tipos de documento:")
        for tipo, count in cargador.df['clase_documento'].value_counts().items():
            print(f"   • {tipo}: {count} chunks")

    # Documentos originales
    if 'nombre_documento' in cargador.df.columns:
        print(f"\n📄 Documentos originales:")
        for doc in cargador.df['nombre_documento'].unique():
            chunks = len(cargador.df[cargador.df['nombre_documento'] == doc])
            tipo = cargador.df[cargador.df['nombre_documento'] == doc]['clase_documento'].iloc[0]
            print(f"   • {doc}: {chunks} chunks ({tipo})")

    # 3. Cargar documentos (necesario para el flujo)
    print(f"\n📄 Cargando documentos en memoria...")
    documentos = cargador.cargar_documentos()
    print(f"   ✅ {len(documentos)} chunks listos")

    print("\n✅ Ingestión completada.")
    return documentos


if __name__ == "__main__":
    documentos = main()