from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
import frontmatter
from typing import Dict, List, Set


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
                # Using this approach for now, may change later
                title = filepath.stem.replace(" ", "_")
                self.metadata[file_id] = {
                    "title": title,
                    "content": post.content
                }

    def generate_embeddings(self):
        for file_id, data in self.metadata.items():
            # TODO: prepend title to content, maybe remove underscores from titles
            # instead of adding underscores
            embedding = self.model.encode(data["content"])
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


def main():
    semantic_search = SemanticSearch(notes_dir="/home/scossar/obsidian/")
    semantic_search.load_documents()
    semantic_search.generate_embeddings()
    results = semantic_search.search("rules of calculus")

    for i in range(len(results)):
        result = results[i]
        for key, value in result.items():
            print(f"{key}:\n{value}")
        print("\n")

    # notes = semantic_search.find_files_by_ids(
    #     [
    #         "zig_13_for_ghostty_20250105_190604_1d7b",
    #         "training_and_testing_on_different_distributions_20250106_162438_3cf7"
    #     ]
    # )
    #
    # for file_id, content in notes.items():
    #     print(f"file_id: {file_id}\ncontent:\n{content}")


if __name__ == "__main__":
    main()


