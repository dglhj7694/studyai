import math
import json
from pathlib import Path


def dot_product(vector_a: list[float], vector_b: list[float]) -> float:
    if len(vector_a) != len(vector_b):
        raise ValueError("두 벡터의 길이가 같아야 합니다.")

    return sum(a * b for a, b in zip(vector_a, vector_b))


def vector_norm(vector: list[float]) -> float:
    return math.sqrt(sum(value ** 2 for value in vector))


def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    dot = dot_product(vector_a, vector_b)
    norm_a = vector_norm(vector_a)
    norm_b = vector_norm(vector_b)

    if norm_a == 0 or norm_b == 0:
        raise ValueError("0 벡터는 코사인 유사도를 계산할 수 없습니다.")

    return dot / (norm_a * norm_b)


def search_by_vector(
    query_vector: list[float],
    documents: list[dict],
    top_k: int = 3,
    min_similarity: float = 0.0,
) -> list[dict]:
    """
    query vector와 가장 비슷한 document를 찾는다.
    """

    results = []

    for document in documents:
        similarity = cosine_similarity(
            query_vector,
            document["vector"],
        )

        result = {
            "id": document["id"],
            "title": document["title"],
            "content": document["content"],
            "vector": document["vector"],
            "similarity": similarity,
        }

        if similarity >= min_similarity:
            results.append(result)

    results.sort(key=lambda x: x["similarity"], reverse=True)

    return results[:top_k]


def save_results(results: list[dict], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    json_text = json.dumps(results, ensure_ascii=False, indent=2)
    path.write_text(json_text, encoding="utf-8")


def print_results(
    query_name: str,
    query_vector: list[float],
    results: list[dict],
    min_similarity: float,
) -> None:
    print("=" * 80)
    print(f"질문: {query_name}")
    print(f"질문 벡터: {query_vector}")
    print(f"최소 유사도: {min_similarity}")
    print("=" * 80)

    if not results:
        print("검색 결과가 없습니다.")
        return

    for index, result in enumerate(results, start=1):
        print(f"\n[{index}] {result['title']}")
        print(f"문서 ID: {result['id']}")
        print(f"유사도: {result['similarity']:.4f}")
        print(f"문서 벡터: {result['vector']}")
        print(f"내용: {result['content']}")


def main():
    # 벡터 의미:
    # [rag, fine_tuning, transformer, evaluation]
    documents = [
        {
            "id": "doc_rag",
            "title": "RAG 개념",
            "content": "RAG는 외부 문서를 검색한 뒤 그 내용을 기반으로 답변하는 방식이다.",
            "vector": [1.0, 0.1, 0.1, 0.3],
        },
        {
            "id": "doc_fine_tuning",
            "title": "Fine-tuning 개념",
            "content": "Fine-tuning은 기존 모델을 특정 목적에 맞게 추가 학습하는 과정이다.",
            "vector": [0.1, 1.0, 0.1, 0.1],
        },
        {
            "id": "doc_transformer",
            "title": "Transformer 개념",
            "content": "Transformer는 Attention을 사용해 문맥 관계를 계산하는 모델 구조이다.",
            "vector": [0.1, 0.1, 1.0, 0.1],
        },
        {
            "id": "doc_evaluation",
            "title": "RAG 평가 개념",
            "content": "RAG 평가는 검색된 문서가 질문에 적절한지, 답변이 근거에 충실한지 확인하는 과정이다.",
            "vector": [0.7, 0.1, 0.1, 1.0],
        },
    ]

    queries = [
        {
            "name": "외부 자료를 참고해서 답변하는 방식",
            "vector": [0.9, 0.2, 0.1, 0.2],
        },
        {
            "name": "모델을 특정 작업에 맞게 추가 학습하는 방법",
            "vector": [0.1, 0.95, 0.1, 0.1],
        },
        {
            "name": "Attention으로 문맥을 이해하는 구조",
            "vector": [0.1, 0.1, 0.95, 0.1],
        },
        {
            "name": "검색 결과와 답변이 좋은지 측정하는 방법",
            "vector": [0.5, 0.1, 0.1, 0.9],
        },
    ]

    min_similarity = 0.0

    all_query_results = []

    for query in queries:
        results = search_by_vector(
            query_vector=query["vector"],
            documents=documents,
            top_k=3,
            min_similarity=min_similarity,
        )

        print_results(
            query_name=query["name"],
            query_vector=query["vector"],
            results=results,
            min_similarity=min_similarity,
        )

        all_query_results.append(
            {
                "query": query["name"],
                "query_vector": query["vector"],
                "results": results,
            }
        )

        print("\n")

    output_path = "phase0/lesson04/outputs/vector_search_results.json"
    save_results(all_query_results, output_path)

    print(f"검색 결과 저장 완료: {output_path}")


if __name__ == "__main__":
    main()