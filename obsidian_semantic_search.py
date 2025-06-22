from pathlib import Path
from sentence_transformers import SentenceTransformer
import frontmatter
from typing import Dict, List, Set
import typer
from rich import print
import chromadb
from chromadb.config import Settings


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
    """Index all markdown files in a directory"""
    semantic_search = SemanticSearch()
    file_path = Path(file_path)
    semantic_search.index_file(filepath=file_path)

@app.command()
def search(query: str):
    """Search indexed documents"""
    semantic_search = SemanticSearch()
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

        print(f"[bold green]{title}[/bold green]\nsimilarity: {similarity}\nlines: {start_line}-{end_line}\n{content}\n\n")

if __name__ == "__main__":
    app()


