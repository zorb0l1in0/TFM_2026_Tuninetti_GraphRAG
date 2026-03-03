import os
from pathlib import Path
from typing import List, Optional, Tuple, Any
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document

from src.extraction.entity_relation_labels_extractor import HybridDocumentAnalyzer




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
            allowed_relationships: Optional[List[str]] = None,
            # Nuovi parametri per scoperta automatica
            auto_discover: bool = False,
            analyzer: Optional[HybridDocumentAnalyzer] = None,
            min_frecuencia: int = 2,
            top_n: int = 15,
            threshold_percentual: float = 0.05  # 👈 5% di default
    ):
        """
        Inicializa el extractor.

        Args:
            threshold_percentual: porcentaje del máximo para filtrar nodos (0.05 = 5%)
        """
        self.verbose = verbose
        self.auto_discover = auto_discover
        self.analyzer = analyzer
        self.min_frecuencia = min_frecuencia
        self.top_n = top_n
        self.threshold_percentual = threshold_percentual

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("❌ OPENAI_API_KEY no encontrada en .env")

        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key
        )

        # Se scoperta automatica è attiva, non passiamo ancora allowed_nodes/relationships
        # perché devono essere scoperti dai documenti
        if not auto_discover:
            self.allowed_nodes = allowed_nodes
            self.allowed_relationships = allowed_relationships
            self.transformer = LLMGraphTransformer(
                llm=self.llm,
                node_properties=node_properties,
                relationship_properties=relationship_properties,
                allowed_nodes=allowed_nodes,
                allowed_relationships=allowed_relationships
            )
            if self.verbose:
                print(f"✅ EntityRelationExtractor inicializado: {model_name}")
                if allowed_nodes:
                    print(f"   📋 Nodos permitidos: {len(allowed_nodes)}")
                if allowed_relationships:
                    print(f"   🔗 Relaciones permitidas: {len(allowed_relationships)}")
        else:
            self.allowed_nodes = None
            self.allowed_relationships = None
            self.transformer = None
            if self.verbose:
                print(f"🔍 EntityRelationExtractor en modo descubrimiento automático")
                if not self.analyzer:
                    print("   📊 Se creará un HybridDocumentAnalyzer para analizar los documentos")

    def _descubre_vocabulario(self, documents: List[Document]) -> Tuple[List[str], List[str]]:
        """
        Scopre nodi e relazioni dai documenti usando HybridDocumentAnalyzer.
        """
        if self.verbose:
            print("\n🔍 Descubriendo vocabulario del dominio con HybridDocumentAnalyzer...")

        # Crea analyzer se non fornito
        if not self.analyzer:
            self.analyzer = HybridDocumentAnalyzer(
                modelo_spacy="es_core_news_lg",
                patron_tabla=["Fila de tabla", "Tabla"]
            )

        # Raccogli tutti i testi dai documenti
        testi_completi = []
        for doc in documents:
            if hasattr(doc, 'page_content'):
                testi_completi.append(doc.page_content)

        # Unisci i testi
        testo_unico = "\n\n".join(testi_completi)

        # Salva temporaneamente
        temp_path = Path("temp_analysis.txt")
        temp_path.write_text(testo_unico, encoding="utf-8")

        # 👈 USA IL THRESHOLD PERCENTUALE
        resultado = self.analyzer.analiza(
            temp_path,
            top_n=self.top_n,
            threshold_percentual=self.threshold_percentual,  # 👈 NUOVO
            min_frecuencia=self.min_frecuencia,
            verbose=self.verbose
        )

        # Pulisci file temporaneo
        temp_path.unlink()

        allowed_nodes = resultado['allowed_nodes']
        allowed_relationships = resultado['allowed_relationships']

        if self.verbose:
            print(f"\n✅ Vocabulario descubierto:")
            print(f"   📋 Nodos ({len(allowed_nodes)}): {allowed_nodes}")
            print(f"   🔗 Relaciones ({len(allowed_relationships)}): {allowed_relationships}")
            if 'threshold_usado' in resultado['estadisticas']:
                print(f"   📊 Threshold usado: {resultado['estadisticas']['threshold_usado']:.1f}")

        return allowed_nodes, allowed_relationships

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

        # Se è in modalità scoperta automatica, prima analizza i documenti
        if self.auto_discover and not self.transformer:
            allowed_nodes, allowed_relationships = self._descubre_vocabulario(documents)

            # Ora inizializza il transformer con il vocabolario scoperto
            self.allowed_nodes = allowed_nodes
            self.allowed_relationships = allowed_relationships
            self.transformer = LLMGraphTransformer(
                llm=self.llm,
                node_properties=["description"],
                relationship_properties=["description"],
                allowed_nodes=allowed_nodes,
                allowed_relationships=allowed_relationships
            )
            if self.verbose:
                print(f"\n✅ Transformer inicializado con vocabulario descubierto")

        if not self.transformer:
            raise ValueError("❌ Transformer no inicializado. Usa auto_discover=True o pasa allowed_nodes/relationships")

        if self.verbose:
            print(f"\n📊 Extrayendo de {len(documents)} docs...")
            if self.allowed_nodes:
                print(f"   Usando vocabulario: {len(self.allowed_nodes)} nodos, {len(self.allowed_relationships)} relaciones")

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

    def extract_with_predefined_vocabulary(
            self,
            documents: List[Document],
            vocabulary_document: Path,
            batch_size: int = 10
    ) -> Tuple[List[Any], int, int]:
        """
        Estrae usando un vocabolario predefinito da un documento specifico.

        Args:
            documents: documenti da analizzare
            vocabulary_document: percorso del documento da cui estrarre il vocabolario
            batch_size: dimensione batch

        Returns:
            (graph_documents, total_entities, total_relationships)
        """
        if self.verbose:
            print(f"\n📖 Usando documento come vocabolario: {vocabulary_document.name}")

        # Crea analyzer se non esiste
        if not self.analyzer:
            self.analyzer = HybridDocumentAnalyzer(
                modelo_spacy="es_core_news_lg",
                patron_tabla=["Fila de tabla", "Tabla"]
            )

        # Analizza il documento vocabolario
        resultado = self.analyzer.analiza(
            vocabulary_document,
            top_n=self.top_n,
            min_frecuencia=self.min_frecuencia,
            verbose=self.verbose
        )

        # Imposta il vocabolario
        self.allowed_nodes = resultado['allowed_nodes']
        self.allowed_relationships = resultado['allowed_relationships']

        # Inizializza transformer
        self.transformer = LLMGraphTransformer(
            llm=self.llm,
            node_properties=["description"],
            relationship_properties=["description"],
            allowed_nodes=self.allowed_nodes,
            allowed_relationships=self.allowed_relationships
        )

        # Estrai dai documenti target
        return self.extract(documents, batch_size)


# Esempio di utilizzo
if __name__ == "__main__":
    from langchain_core.documents import Document

    # Esempio 1: Scoperta automatica
    extractor1 = EntityRelationExtractor(
        model_name="gpt-4o-mini",
        verbose=True,
        auto_discover=True,
        top_n=15,
        min_frecuencia=2
    )

    # Documenti da analizzare
    docs = [
        Document(page_content="Texto del documento da analizzare..."),
        Document(page_content="Altro documento...")
    ]

    graph_docs, entities, rels = extractor1.extract(docs)
    print(f"Estratti: {entities} entità, {rels} relazioni")

    # Esempio 2: Usa un documento come vocabolario e un altro come target
    extractor2 = EntityRelationExtractor(
        model_name="gpt-4o-mini",
        verbose=True,
        auto_discover=False  # Non scopre automaticamente
    )

    graph_docs2, entities2, rels2 = extractor2.extract_with_predefined_vocabulary(
        documents=docs,  # documenti target
        vocabulary_document=Path("../../data/raw/decreto_precios_publicos_universidades_publicas_canarias.md"),
        batch_size=10
    )