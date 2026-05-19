from pathlib import Path
import json

import chromadb
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "lesson06_rag_chunks"
CHROMA_PATH = "phase0/lesson06/chroma_db"


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
    return collection


def search_chroma(
    query: str,
    n_results: int = 5,
) -> dict:
    print(f"Embedding model 로딩 중: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    ).tolist()

    collection = get_collection()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    return results


def format_results(raw_results: dict) -> list[dict]:
    """
    Chroma query 결과는 query batch 기준으로 묶여 있다.
    지금은 query 1개만 넣으므로 [0] 번째 묶음을 꺼낸다.
    """

    formatted = []

    ids = raw_results["ids"][0]
    documents = raw_results["documents"][0]
    metadatas = raw_results["metadatas"][0]
    distances = raw_results["distances"][0]

    for chunk_id, document, metadata, distance in zip(
        ids,
        documents,
        metadatas,
        distances,
    ):
        formatted.append(
            {
                "chunk_id": chunk_id,
                "distance": distance,
                "file_name": metadata["file_name"],
                "category": metadata["category"],
                "chunk_index": metadata["chunk_index"],
                "start_char": metadata["start_char"],
                "end_char": metadata["end_char"],
                "preview": metadata["preview"],
                "content": document,
            }
        )

    return formatted


def save_results(results: list[dict], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    json_text = json.dumps(results, ensure_ascii=False, indent=2)
    path.write_text(json_text, encoding="utf-8")


def print_results(query: str, results: list[dict]) -> None:
    print("=" * 80)
    print(f"질문: {query}")
    print(f"검색 결과 수: {len(results)}")
    print("=" * 80)

    if not results:
        print("검색 결과가 없습니다.")
        return

    for index, result in enumerate(results, start=1):
        print(f"\n[{index}] {result['file_name']} / chunk {result['chunk_index']}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Distance: {result['distance']:.4f}")
        print(f"Category: {result['category']}")
        print(f"문자 범위: {result['start_char']} ~ {result['end_char']}")
        print(f"미리보기: {result['preview']}")
        print("\n본문:")
        print(result["content"][:400])


def main():
    output_path = "phase0/lesson06/outputs/chroma_search_results.json"

    query = input("질문을 입력하세요: ").strip()

    if not query:
        print("질문이 비어 있습니다.")
        return

    raw_results = search_chroma(
        query=query,
        n_results=5,
    )

    results = format_results(raw_results)

    print_results(query, results)
    save_results(results, output_path)

    print("\n검색 결과 저장 완료")
    print(f"저장 위치: {output_path}")


if __name__ == "__main__":
    main()