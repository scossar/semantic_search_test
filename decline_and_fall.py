from pathlib import Path
from chromadb.api.types import normalize_embeddings
import nltk
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import numpy as np
nltk.download("punkt")
from nltk.tokenize import sent_tokenize, word_tokenize

import typer
from rich.console import Console
from rich import print
from rich.panel import Panel
from rich.text import Text
# from rich.columns import Columns
from rich.box import ROUNDED, MINIMAL

#######
# this is a mess
# notes:
# splitting_text_into_sentences_for_embeddings.md

class DeclineSearch:
    def __init__(self, model_name: str="all-mpnet-base-v2",
                 persist_directory: str="./origin_db"):
        self.model = SentenceTransformer(model_name)
        self.chroma_client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name="decline_chunks",
            metadata={"hnsw:space": "cosine"}
        )
        self.paragraphs = []
        self.chunks = []

    def set_paragraphs(self, filepath: Path):
        with open(filepath, "r", encoding="utf-8") as file:
            text = file.read()
        self.paragraphs = text.split('\n\n')

    def chunk_paragraph(self, paragraph, max_tokens=350):
        sentences = sent_tokenize(paragraph)
        # chunks = []
        current_chunk = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = len(word_tokenize(sentence))

            if current_tokens + sentence_tokens > max_tokens and current_chunk:
                self.chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens

        if current_chunk:
            self.chunks.append(" ".join(current_chunk))

    # def chunk_paragraphs(self):
    #     self.set_paragraphs(Path("./history_of_the_decline_and_fall_of_the_roman_empire_volume_1.txt"))
    #     for paragraph in self.paragraphs:
    #         self.chunk_paragraph(paragraph)

    def index_document(self):
        self.set_paragraphs(Path("./on_the_origin_of_species.txt"))
        for paragraph in self.paragraphs:
            self.chunk_paragraph(paragraph)

        print(f"paragraph chunks set: {len(self.chunks)}")

        ids = []
        embeddings = []
        documents = []

        for i, chunk in enumerate(self.chunks):
            chunk_id = f"df_{i}"
            ids.append(chunk_id)
            documents.append(chunk)

            embedding = self.model.encode(
                chunk,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            embeddings.append(embedding.tolist())
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents
            )

    def search(self, query: str, top_k: int=5):
        query_embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["metadatas", "documents", "distances"]
        )
        formatted_results = []

        if not (results["metadatas"] and results["documents"] and results["distances"]):
            return []

        for i in range(len(results['ids'][0])):
            formatted_results.append({
                "similarity": 1 - results['distances'][0][i],  # Convert distance to similarity
                "metadata": {
                    # **results['metadatas'][0][i],
                    "content": results['documents'][0][i]
                }
            })

        return formatted_results

app = typer.Typer()
console = Console()

@app.command()
def index():
    DeclineSearch().index_document()
    print("document indexed")

@app.command()
def search(query: str, top_k: int = 5):
    """Search indexed documents"""
    results = DeclineSearch().search(query, top_k)

    console.print(f"\n[bold cyan]Search results for:[/bold cyan] [yellow]{query}[/yellow]\n")

    for result in results:
        print(f"\n\nsimilarity: {result['similarity']}")
        print(f"\n{result['metadata']['content']}")


if __name__ == "__main__":
    app()
