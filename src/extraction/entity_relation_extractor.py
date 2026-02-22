"""
Extractor de entidades y relaciones para GraphRAG.
Inspirado en el paper Microsoft "From Local to Global".
"""

import os
from typing import List, Optional, Tuple, Any
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document

load_dotenv()


class EntityRelationExtractor:
    """
    Extractor de entidades y relaciones usando LLM.
    Extrae nodos, relaciones y descripciones del texto.
    NO guarda en Neo4j, solo extrae.
    """

    def __init__(
            self,
            model_name: str = "gpt-4o-mini",
            temperature: float = 0,
            verbose: bool = True,
            node_properties: List[str] = ["description"],
            relationship_properties: List[str] = ["description"],
            allowed_nodes: Optional[List[str]] = None,
            allowed_relationships: Optional[List[str]] = None
    ):
        """
        Inicializa el extractor.
        """
        self.verbose = verbose

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("❌ OPENAI_API_KEY no encontrada en .env")

        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key
        )

        self.transformer = LLMGraphTransformer(
            llm=self.llm,
            node_properties=node_properties,
            relationship_properties=relationship_properties,
            allowed_nodes=allowed_nodes,
            allowed_relationships=allowed_relationships
        )

        if self.verbose:
            print(f"✅ EntityRelationExtractor inicializado: {model_name}")

    def extract(
            self,
            documents: List[Document],
            batch_size: int = 10
    ) -> Tuple[List[Any], int, int]:
        """
        Extrae entidades y relaciones de los documentos.

        Returns:
            (graph_documents, total_entities, total_relationships)
        """
        if not documents:
            return [], 0, 0

        if self.verbose:
            print(f"\n📊 Extrayendo de {len(documents)} docs...")

        all_graph_documents = []
        total_entities = 0
        total_relationships = 0

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]

            try:
                graph_documents = self.transformer.convert_to_graph_documents(batch)

                entities = sum(len(gd.nodes) for gd in graph_documents)
                relationships = sum(len(gd.relationships) for gd in graph_documents)

                if self.verbose:
                    print(f"   Lote {i // batch_size + 1}: {entities} ent, {relationships} rel")

                all_graph_documents.extend(graph_documents)
                total_entities += entities
                total_relationships += relationships

            except Exception as e:
                if self.verbose:
                    print(f"      ❌ Error lote: {e}")

        return all_graph_documents, total_entities, total_relationships