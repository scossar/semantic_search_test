from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
import frontmatter
from typing import Dict, List, Set, TypedDict
# from sentence_transformers.util import normalize_embeddings
import typer
from rich import print
import chromadb
from chromadb.config import Settings

# notes:
# - generate_vector_embeddings_for_local_files.md
# - python_typing_library.md
# - generating_embeddings_from_markdown_posts.md
# NOTE: the code is a mess. contains unused methods, etc. it works though

class ChunkMetadata(TypedDict):
    start_line: int
    end_line: int
    text: str

class DocumentMetadata(TypedDict):
    title: str
    chunks: List[ChunkMetadata]

class ChunkEmbedding(TypedDict):
    start_line: int
    end_line: int
    embedding: np.ndarray


class SemanticSearch:
    def __init__(self, notes_dir: str, model_name: str = "all-mpnet-base-v2",
                 persist_directory: str = "./chroma_db"):
        self.notes_dir = Path(notes_dir)
        self.skip_dirs: Set[str] = {
            "node_modules",
            ".git",
            ".obsidian",
            "__pycache__",
            "venv",
            ".venv"
        }
        self.model = SentenceTransformer(model_name)
        self.chroma_client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        self.collection = self.chroma_client.get_or_create_collection(
            name="markdown_chunks",
            metadata={"hnsw:space": "cosine"}
        )
        self.embeddings: Dict[str, List[ChunkEmbedding]] = {}
        self.metadata: Dict[str, DocumentMetadata] = {}

    def should_process_file(self, filepath: Path) -> bool:
        if any(part.startswith(".") for part in filepath.parts):
            return False
        if any(skip_dir in filepath.parts for skip_dir in self.skip_dirs):
            return False
        if filepath.suffix.lower() not in (".md", ".markdown"):
            return False
        return True

    def sliding_window_chunks(self, content: str, window_lines: int = 10, stride: int = 5):
        lines = content.split('\n')
        chunks = []

        for i in range(0, len(lines), stride):
            end = min(i + window_lines, len(lines))
            chunk_text = '\n'.join(lines[i:end])
            if chunk_text.strip():  # skip empty
                chunks.append({
                    "text": chunk_text,
                    "start_line": i,
                    "end_line": end - 1
                })

        return chunks


    def index_documents(self):
        """Index all documents - only needs to be run once or when files change"""
        for filepath in self.notes_dir.rglob("*"):
            if not self.should_process_file(filepath):
                continue

            post = frontmatter.load(str(filepath))
            file_id = post.get("file_id")

            if isinstance(file_id, str):
                chunks = self.sliding_window_chunks(post.content)
                title = filepath.stem.replace("_", " ")

                ids = []
                embeddings = []
                metadatas = []
                documents = []

                for i, chunk in enumerate(chunks):
                    chunk_id = f"{file_id}_{i}"
                    embedding_text = f"{title}\n{chunk['text']}"

                    ids.append(chunk_id)
                    documents.append(chunk['text'])
                    metadatas.append({
                        "file_id": file_id,
                        "title": title,
                        "start_line": chunk['start_line'],
                        "end_line": chunk['end_line']
                    })

                    embedding = self.model.encode(
                        embedding_text,
                        convert_to_numpy=True,
                        normalize_embeddings=True
                    )
                    embeddings.append(embedding.tolist())

                if ids:
                    self.collection.add(
                        ids=ids,
                        embeddings=embeddings,
                        metadatas=metadatas,
                        documents=documents
                    )

    def load_documents(self):
        for filepath in self.notes_dir.rglob("*"):
            if not self.should_process_file(filepath):
                continue

            post = frontmatter.load(str(filepath))
            file_id = post.get("file_id")
            if isinstance(file_id, str):
                title = filepath.stem.replace("_", " ")
                chunks = self.sliding_window_chunks(post.content)
                self.metadata[file_id] = {
                    "title": title,
                    "chunks": chunks
                }
    
    def generate_embeddings(self):
        for file_id, doc_metadata in self.metadata.items():
            chunk_embeddings = []

            for chunk in doc_metadata["chunks"]:
                embedding_text = f"{doc_metadata['title']}\n{chunk['text']}"
                embedding = self.model.encode(
                    embedding_text,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
                chunk_embeddings.append({
                    "start_line": chunk["start_line"],
                    "end_line": chunk["end_line"],
                    "embedding": embedding
                })

            self.embeddings[file_id] = chunk_embeddings

    def search(self, query: str, top_k: int = 5):
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
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                "similarity": 1 - results['distances'][0][i],  # Convert distance to similarity
                "metadata": {
                    **results['metadatas'][0][i],
                    "content": results['documents'][0][i]
                }
            })

        return formatted_results

    def find_files_by_ids(self, file_ids: List[str]) -> Dict[str, str]:
        results = {}

        for filepath in self.notes_dir.rglob("*"):
            if not self.should_process_file(filepath):
                continue

            post = frontmatter.load(str(filepath))
            file_id = post.get("file_id")
            if file_id and file_id in file_ids:
                results[file_id] = post.content

        return results


app = typer.Typer()

@app.command()
def index(notes_dir: str):
    """Index all documents in notes_dir"""
    semantic_search = SemanticSearch(notes_dir=notes_dir)
    semantic_search.index_documents()
    print("Indexing complete!")

@app.command()
def search(notes_dir: str, query: str):
    """Search indexed documents"""
    semantic_search = SemanticSearch(notes_dir=notes_dir)
    results = semantic_search.search(query)

    print("\n")
    for i in range(len(results)):
        result = results[i]
        similarity = result["similarity"]
        metadata = result["metadata"]
        title = metadata["title"]
        content = metadata["content"]
        start_line = metadata["start_line"]
        end_line = metadata["end_line"]

        print(f"[bold green]{title}[/bold green]\nsimilarity: {similarity}\nstart_line: {start_line}\n{content}\n\n")


if __name__ == "__main__":
    app()


