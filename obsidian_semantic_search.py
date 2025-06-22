from pathlib import Path
from chromadb.api.types import normalize_embeddings
from sentence_transformers import SentenceTransformer
import numpy as np
import frontmatter
from typing import Dict, List, Set, TypedDict
import typer
from rich import print
import chromadb
from chromadb.config import Settings

# notes:
# - generate_vector_embeddings_for_local_files.md
# - python_typing_library.md
# - generating_embeddings_from_markdown_posts.md
# - getting_started_with_chromadb.md

# class ChunkMetadata(TypedDict):
#     start_line: int
#     end_line: int
#     text: str
#
# class DocumentMetadata(TypedDict):
#     title: str
#     chunks: List[ChunkMetadata]
#
# class ChunkEmbedding(TypedDict):
#     start_line: int
#     end_line: int
#     embedding: np.ndarray


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
            settings=Settings(anonymized_telemetry=False)  # chroma collects usage info: set to True to disable
        )

        # no embedding function is supplied, so chroma will use sentence_transformer
        self.collection = self.chroma_client.get_or_create_collection(
            name="markdown_chunks",
            # I think these are the default values, the Chroma docs aren't great. It seems that metadata can be# used for configuration.
            metadata={"hnsw:space": "cosine"}
        )
        # self.embeddings: Dict[str, List[ChunkEmbedding]] = {}
        # self.metadata: Dict[str, DocumentMetadata] = {}


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

    def index_documents(self, force_all=False, file_path=None):
        """
        Index documents with update detection

        Args:
            force_all: if True, re-index everything
            file_path: if provided, only index the specific file
        """
        if file_path:
            self._index_file(Path(file_path))
            print(f"indexed [bold green]{file_path}[/bold green]\n")
        else:
            for path in self.notes_dir.rglob("*"):
                if not force_all:
                    if not self._should_process_file(path):
                        continue
                    if self._is_up_to_date(path):
                        print(f"skipping indexing for [bold green]{path}[/bold green]\nFile up to date")
                        continue
                self._index_file(path)
                print(f"indexed [bold green]{path}[/bold green]\n")

    def _index_file(self, filepath: Path):
        """Index a single document"""

        post = frontmatter.load(str(filepath))
        file_id = post.get("file_id")

        if isinstance(file_id, str):
            file_mtime = filepath.stat().st_mtime
            chunks = self.sliding_window_chunks(post.content)
            title = filepath.stem.replace("_", " ")  # this can be improved

            ids = []
            embeddings = []
            metadatas = []
            documents = []

            for i, chunk in enumerate(chunks):
                chunk_id = f"{file_id}_{i}"
                # embedding_text = chunk["text"]  # previously I was prepending the title

                ids.append(chunk_id)
                documents.append(chunk["text"])
                metadatas.append({
                    "file_id": file_id,
                    "title": title,
                    "start_line": chunk["start_line"],
                    "end_line": chunk["end_line"],
                    "indexed_at": file_mtime
                })

                embedding = self.model.encode(
                    chunk["text"],
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                embeddings.append(embedding.tolist())

                if ids:
                    self.collection.upsert(
                        ids=ids,
                        embeddings=embeddings,
                        metadatas=metadatas,
                        documents=documents
                    )

    def _is_up_to_date(self, filepath: Path) -> bool:
        """Check if a file needs re-indexing based on modification time"""
        post = frontmatter.load(str(filepath));
        file_id = post.get("file_id")

        if not isinstance(file_id, str):
            return False  # probably need a different check here to handle case of no file_id frontmatter
        
        file_mtime = filepath.stat().st_mtime
        existing = self.collection.get(
            where={"file_id": file_id},
            limit=1,
            include=["metadatas"]
        ) 

        if not existing["ids"]:  # no chunks found, needs to be indexed
            return False

        if not existing["metadatas"]:  # "metadatas" is an Optional property
            return False

        indexed_at = existing["metadatas"][0].get("indexed_at", 0)

        if not isinstance(indexed_at, (int, float)):
            return False  # invalid timestamp

        return file_mtime <= indexed_at

    def _should_process_file(self, filepath: Path) -> bool:
        if any(part.startswith(".") for part in filepath.parts):
            return False
        if any(skip_dir in filepath.parts for skip_dir in self.skip_dirs):
            return False
        if filepath.suffix.lower() not in (".md", ".markdown"):
            return False
        return True

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


        if not (results["metadatas"] and results["documents"] and results["distances"]):
            return []

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
            if not self._should_process_file(filepath):
                continue

            post = frontmatter.load(str(filepath))
            file_id = post.get("file_id")
            if file_id and file_id in file_ids:
                results[file_id] = post.content

        return results


app = typer.Typer()

@app.command()
def index_directory(notes_dir: str):
    """Index all markdown files in a directory"""
    semantic_search = SemanticSearch(notes_dir=notes_dir)
    semantic_search.index_documents()

@app.command()
def index_file(notes_dir: str, file_path: str):
    """Index all markdown files in a directory"""
    semantic_search = SemanticSearch(notes_dir=notes_dir)
    semantic_search.index_documents(file_path=file_path)

@app.command()
def search(notes_dir: str, query: str):
    """Search indexed documents"""
    semantic_search = SemanticSearch(notes_dir=notes_dir)  # the notes_dir isn't used to limit the scope of the search
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


