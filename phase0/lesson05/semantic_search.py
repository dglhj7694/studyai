from pathlib import Path
import json

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_embedding_index(index_path: str) -> list[dict]:
    path = Path(index_path)

    if not path.exists():
        raise FileNotFoundError(f"Embedding index 파일을 찾을 수 없습니다: {index_path}")

    json_text = path.read_text(encoding="utf-8")
    return json.loads(json_text)


def cosine_similarity_normalized(vector_a: list[float], vector_b: list[float]) -> float:
    """
    normalize_embeddings=True로 만든 벡터는 이미 길이가 1에 가깝다.
    그래서 내적이 곧 cosine similarity처럼 동작한다.
    """

    a = np.array(vector_a)
    b = np.array(vector_b)

    return float(np.dot(a, b))


def semantic_search(
    query: str,
    indexed_chunks: list[dict],
    model: SentenceTransformer,
    top_k: int = 5,
    min_similarity: float = 0.0,
) -> list[dict]:
    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    ).tolist()

    results = []

    for chunk in indexed_chunks:
        similarity = cosine_similarity_normalized(
            query_embedding,
            chunk["embedding"],
        )

        if similarity >= min_similarity:
            result = {
                "chunk_id": chunk["id"],
                "document_id": chunk["document_id"],
                "similarity": similarity,
                "file_name": chunk["metadata"]["file_name"],
                "category": chunk["metadata"]["category"],
                "chunk_index": chunk["metadata"]["chunk_index"],
                "start_char": chunk["metadata"]["start_char"],
                "end_char": chunk["metadata"]["end_char"],
                "preview": chunk["metadata"]["preview"],
                "content": chunk["content"],
            }

            results.append(result)

    results.sort(key=lambda x: x["similarity"], reverse=True)

    return results[:top_k]


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
        print(f"Similarity: {result['similarity']:.4f}")
        print(f"Category: {result['category']}")
        print(f"문자 범위: {result['start_char']} ~ {result['end_char']}")
        print(f"미리보기: {result['preview']}")
        print("\n본문:")
        print(result["content"][:400])


def main():
    index_path = "phase0/lesson05/outputs/embedding_index.json"
    output_path = "phase0/lesson05/outputs/semantic_search_results.json"

    indexed_chunks = load_embedding_index(index_path)

    print(f"Embedding model 로딩 중: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    query = input("질문을 입력하세요: ").strip()

    if not query:
        print("질문이 비어 있습니다.")
        return

    results = semantic_search(
        query=query,
        indexed_chunks=indexed_chunks,
        model=model,
        top_k=5,
        min_similarity=0.0,
    )

    print_results(query, results)
    save_results(results, output_path)

    print("\n검색 결과 저장 완료")
    print(f"저장 위치: {output_path}")


if __name__ == "__main__":
    main()