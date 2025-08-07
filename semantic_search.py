from pathlib import Path
from sentence_transformers import SentenceTransformer
import frontmatter  # package: python-frontmatter
from typing import Dict, List, Set
import typer
from rich import print
import chromadb
from chromadb.config import Settings
import numpy as np

from rich.console import Console
from rich.panel import Panel
# from rich.table import Table
from rich.syntax import Syntax
from rich.text import Text
# from rich.columns import Columns
from rich.box import ROUNDED, MINIMAL

import json

console = Console()
###############################################################################
# notes:
# - generating_embeddings_from_markdown_posts.md
# - semantic_search_and_vector_databases.md
# - getting_started_with_chromadb.md
# - handling_chunked_data_returned_from_chroma.md

class SemanticSearch:
    def __init__(self, model_name: str = "all-mpnet-base-v2",
                 persist_directory: str = "./chroma_db"):
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
            settings=Settings(anonymized_telemetry=False)  # chroma collects usage info (?)
        )

        # no embedding function is supplied, so chroma will use sentence_transformer
        self.collection = self.chroma_client.get_or_create_collection(
            name="markdown_chunks",
            # I think these are the default values, the Chroma docs aren't great. It seems that metadata can be used for configuration.
            metadata={"hnsw:space": "cosine"}
        )

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

    def index_documents(self, notes_dir: str, force_all=False):
        """
        Index documents with update detection

        Args:
            notes_dir: the directory to index
            force_all: if True, re-index everything (not yet implemented)
        """

        for path in Path(notes_dir).rglob("*"):
            if not force_all:
                if not self._should_process_file(path):
                    continue
                if self._is_up_to_date(path):
                    print(f"Skipping indexing for [bold green]{path}[/bold green]\nFile up to date")
                    continue
            self.index_file(path)
            print(f"Indexed [bold green]{path}[/bold green]\n")

    def index_file(self, filepath: Path):
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

                ids.append(chunk_id)
                documents.append(chunk["text"])
                metadatas.append({
                    "file_id": file_id,
                    "title": title,
                    "start_line": chunk["start_line"],
                    "end_line": chunk["end_line"],
                    "indexed_at": file_mtime,
                    "file_path": str(filepath)
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

        # add 1 second tolerance to deal with rounding issues
        return file_mtime <= indexed_at + 1.0

    def _should_process_file(self, filepath: Path) -> bool:
        if any(part.startswith(".") for part in filepath.parts):
            return False
        if any(skip_dir in filepath.parts for skip_dir in self.skip_dirs):
            return False
        if filepath.suffix.lower() not in (".md", ".markdown"):
            return False
        return True

    def search(self, query: str, top_k: int = 5, chunks_per_file: int = 3):
        query_embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k * 3,  # not quite right
            include=["metadatas", "documents", "distances"]
        )

        if not (results["metadatas"] and results["documents"] and results["distances"]):
            return []

        file_groups = {}
        for i in range(len(results["ids"][0])):
            file_id = results["metadatas"][0][i]["file_id"]
            if file_id not in file_groups:
                file_groups[file_id] = []

            file_groups[file_id].append({
                "similarity": 1 - results["distances"][0][i],
                "metadata": {
                    **results["metadatas"][0][i],
                    "content": results["documents"][0][i]
                }
            })
        # sort files by best chunk's similarity
        sorted_files = sorted(
            file_groups.items(),
            key=lambda x: x[1][0]["similarity"],
            reverse=True
        )[:top_k]

        formatted_results = []
        for file_id, chunks in sorted_files:
            formatted_results.append({
                "file_id": file_id,
                "file_path": chunks[0]["metadata"]["file_path"],
                "title": chunks[0]["metadata"]["title"],
                "best_similarity": chunks[0]["similarity"],
                "chunks": chunks[:chunks_per_file]
            })

        return formatted_results

    def search_center(self, top_k: int=15):
        all_data = self.collection.get(include=["embeddings"])
        embeddings = np.array(all_data["embeddings"])
        centroid = embeddings.mean(axis=0)

        results = self.collection.query(
            query_embeddings=[centroid.tolist()],
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

    def dump_json(self):
        all_data = self.collection.get(include=["embeddings"])
        embeddings = np.array(all_data["embeddings"]).tolist()
        file_path = "embeddings.json"
        with open(file_path, "w") as json_file:
            json.dump(embeddings, json_file, indent=4)

    def find_files_by_ids(self, notes_dir: str,  file_ids: List[str]) -> Dict[str, str]:
        results = {}

        for filepath in Path(notes_dir).rglob("*"):
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
    semantic_search = SemanticSearch()
    semantic_search.index_documents(notes_dir)

@app.command()
def index_file(file_path: Path):
    """Index a given file"""
    semantic_search = SemanticSearch()
    file_path = Path(file_path)
    semantic_search.index_file(filepath=file_path)

# e.g: python obsidian_semantic_search search "installing Postgres" --top-k=2
@app.command()
def search(query: str, top_k: int = 5, chunks_per_file: int = 3):
    """Search indexed documents"""
    results = SemanticSearch().search(query, top_k, chunks_per_file)

    console.print(f"\n[bold cyan]Search results for:[/bold cyan] [yellow]{query}[/yellow]\n")

    for i, result in enumerate(results, 1):
        header = Text()
        header.append(f"{i}. ", style="bold cyan")
        header.append(result['title'], style="bold green")

        info_line = f"[dim]{result['file_path']} • {result['best_similarity']:.3f} similarity[/dim]"

        file_panel = Panel(
            info_line,
            title=header,
            title_align="left",
            border_style="green",
            box=ROUNDED,
            padding=(0, 1)
        )
        console.print(file_panel)

        for j, chunk in enumerate(result["chunks"], 1):
            chunk_header = f"[cyan]Chunk {j}/{len(result['chunks'])}[/cyan] • [dim]similarity: {chunk['similarity']:.3f} • lines: {chunk['metadata']['start_line']}-{chunk['metadata']['end_line']}[/dim]"

            content = chunk['metadata']['content']
            content_display = Text(content, style="dim white")
            chunk_panel = Panel(
                content_display,
                title=chunk_header,
                title_align="left",
                border_style="blue",
                box=MINIMAL,
                padding=(0, 2)
            )

            console.print(chunk_panel, width=console.width - 4)

        console.print()

@app.command()
def search_basic(query: str, top_k: int = 5, chunks_per_file: int = 3):
    """Search indexed documents"""
    results = SemanticSearch().search(query, top_k, chunks_per_file)

    for i in range(len(results)):
        result = results[i]
        print(f"\n[bold green underline]{result['title']}[/bold green underline]")
        print(f"path: {result['file_path']}")
        print(f"file_id: {result['file_id']}")
        print(f"best similarity: {result['best_similarity']}")
        chunks = result["chunks"]
        print(f"chunks: {len(chunks)}")
        for chunk in chunks:
            print(f"\nsimilarity: {chunk['similarity']}")
            metadata = chunk["metadata"]
            print(f"lines: {metadata['start_line']} - {metadata['end_line']}")
            print(f"content:\n{metadata['content']}")

@app.command()
def search_center():
    results = SemanticSearch().search_center()
    for i in range(len(results)):
        result = results[i]
        similarity = result["similarity"]
        metadata = result["metadata"]
        title = metadata["title"]
        content = metadata["content"]
        start_line = metadata["start_line"]
        end_line = metadata["end_line"]
        file_path = metadata["file_path"]

        print(f"[bold green]{title}[/bold green]\nsimilarity: {similarity}\nlines: {start_line}-{end_line}\n")
        print(f"path: [blue]{file_path}[/blue]\n")
        print(f"content:\n{content}\n\n")

@app.command()
def dump_json():
    SemanticSearch().dump_json()

if __name__ == "__main__":
    app()


