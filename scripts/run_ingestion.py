#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de ingestión de datos - Construye el grafo desde CSV
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.ingestion.loader import CSVLoader
from src.ingestion.graphbuilder import GraphBuilder


def main():
    """Función principal de ingestión"""

    print("\n" + "=" * 60)
    print("🚀 INGESTIÓN DE DATOS - GRAPHRAG")
    print("=" * 60)

    # 1. Cargar CSV
    NOMBRE_CSV = "chunks.csv"

    print("\n📂 PASO 1: CARGAR CSV")
    print("-" * 30)

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

    # 3. Cargar documentos en memoria
    print(f"\n📄 Cargando documentos en memoria...")
    documentos = cargador.cargar_documentos()
    print(f"   ✅ {len(documentos)} chunks listos")

    # 4. Construir grafo
    print("\n" + "=" * 60)
    print("🕸️  PASO 2: CONSTRUIR GRAFO")
    print("=" * 60)

    try:
        # Inicializar builder
        builder = GraphBuilder(
            model_name="gpt-4o-mini",  # Modelo económico
            batch_size=5,  # Procesar de 5 en 5
            verbose=True
        )

        # Construir grafo (pasar documentos)
        entidades, relaciones = builder.construir(
            documentos=documentos,
            limpiar=True  # Borrar grafo anterior
        )

        # Verificación rápida
        print("\n" + "=" * 60)
        print("🔍 VERIFICACIÓN")
        print("=" * 60)

        # Consulta de ejemplo
        result = builder.consultar("MATCH (n) RETURN count(n) as total")
        if result:
            print(f"\n📊 Nodos en Neo4j: {result[0]['total']}")

        print("\n🌐 Puedes explorar el grafo en:")
        print("   http://localhost:7474")

    except Exception as e:
        print(f"\n❌ Error construyendo grafo: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 60)
    print("✅ INGESTIÓN COMPLETADA")
    print("=" * 60)
    print(f"📊 {len(documentos)} chunks procesados")
    print(f"🕸️  {entidades} entidades creadas")
    print(f"🔗 {relaciones} relaciones creadas")


if __name__ == "__main__":
    main()