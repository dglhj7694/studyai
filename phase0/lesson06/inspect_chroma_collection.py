import chromadb


COLLECTION_NAME = "lesson06_rag_chunks"
CHROMA_PATH = "phase0/lesson06/chroma_db"


def main():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)

    print("=" * 80)
    print("Chroma Collection 정보")
    print("=" * 80)
    print(f"Collection name: {collection.name}")
    print(f"Count: {collection.count()}")

    print("\n샘플 데이터:")
    result = collection.peek(limit=3)

    ids = result["ids"]
    documents = result["documents"]
    metadatas = result["metadatas"]

    for id_, document, metadata in zip(ids, documents, metadatas):
        print("-" * 80)
        print(f"ID: {id_}")
        print(f"File: {metadata['file_name']}")
        print(f"Category: {metadata['category']}")
        print(f"Preview: {metadata['preview']}")
        print(f"Document: {document[:200]}")


if __name__ == "__main__":
    main()