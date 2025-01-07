from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
import frontmatter
from typing import Dict, List, Set
import typer
from rich import print


class SemanticSearch:
    def __init__(self, notes_dir: str, model_name: str = "all-MiniLM-L6-v2"):
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
        self.embeddings: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict] = {}

    def should_process_file(self, filepath: Path) -> bool:
        if any(part.startswith(".") for part in filepath.parts):
            return False
        if any(skip_dir in filepath.parts for skip_dir in self.skip_dirs):
            return False
        if filepath.suffix.lower() not in (".md", ".markdown"):
            return False
        return True

    def load_documents(self):
        for filepath in self.notes_dir.rglob("*"):
            if not self.should_process_file(filepath):
                continue

            post = frontmatter.load(filepath)
            file_id = post.get("file_id")
            if file_id:
                title = filepath.stem.replace("_", " ")
                self.metadata[file_id] = {
                    "title": title,
                    "content": post.content
                }

    def generate_embeddings(self):
        for file_id, data in self.metadata.items():
            # TODO: prepend title to content, maybe remove underscores from titles
            # instead of adding underscores
            embedding = self.model.encode(f"{data['title']}\n{data['content']}")
            self.embeddings[file_id] = embedding

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        query_embedding = self.model.encode(query)

        similarities = {}
        for file_id, doc_embedding in self.embeddings.items():
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            similarities[file_id] = similarity

        top_results = sorted(similarities.items(),
                             key=lambda x: x[1],
                             reverse=True)[:top_k]

        return [
            {
                "file_id": file_id,
                "similarity": score,
                "metadata": self.metadata[file_id]
            }
            for file_id, score in top_results
        ]

    def find_files_by_ids(self, file_ids: List[str]) -> Dict[str, str]:
        results = {}

        for filepath in self.notes_dir.rglob("*"):
            if not self.should_process_file(filepath):
                continue

            post = frontmatter.load(filepath)
            file_id = post.get("file_id")
            if file_id and file_id in file_ids:
                results[file_id] = post.content

        return results


app = typer.Typer()


@app.command()
def main(notes_dir: str, query: str):
    """
    Search notes_dir for query
    """
    path = Path(notes_dir)

    if not path.exists():
        typer.echo(f"Error: Directory '{notes_dir}' does not exist")
        raise typer.Exit(1)

    if not path.is_dir():
        typer.echo(f"Error: '{notes_dir}' is not a directory")
        raise typer.Exit(1)

    semantic_search = SemanticSearch(notes_dir=notes_dir)
    semantic_search.load_documents()
    semantic_search.generate_embeddings()
    results = semantic_search.search(query)

    print("\n")
    for i in range(len(results)):
        result = results[i]
        similarity = result["similarity"]
        metadata = result["metadata"]
        title = metadata["title"]
        content = metadata["content"]

        print(f"[bold green]{title}[/bold green]\nsimilarity: {similarity}\n{content}\n\n")


if __name__ == "__main__":
    app()


