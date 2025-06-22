import chromadb

chroma_client = chromadb.Client();

collection = chroma_client.create_collection(name="my_collection")

collection.add(
    documents=[
        "This is a document about bicycles",
        "This is a document about fruit"
    ],
    ids=["id1", "id2"]
)

results = collection.query(
    query_texts=["This is a query about bicycles"],
    n_results=2
)

print(results)
