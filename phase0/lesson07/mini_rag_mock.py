from pathlib import Path
import json

import chromadb
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "lesson06_rag_chunks"
CHROMA_PATH = "phase0/lesson06/chroma_db"


def get_collection():
    """
    lesson06에서 만든 Chroma collection을 불러온다.
    """

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)

    return collection


def retrieve_relevant_chunks(
    query: str,
    n_results: int = 5,
) -> list[dict]:
    """
    사용자 질문과 관련 있는 chunk를 Chroma에서 검색한다.
    """

    print(f"Embedding model 로딩 중: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    ).tolist()

    collection = get_collection()

    raw_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    results = []

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
        results.append(
            {
                "chunk_id": chunk_id,
                "document_id": metadata["document_id"],
                "distance": distance,
                "content": document,
                "metadata": {
                    "file_name": metadata["file_name"],
                    "category": metadata["category"],
                    "chunk_index": metadata["chunk_index"],
                    "start_char": metadata["start_char"],
                    "end_char": metadata["end_char"],
                    "preview": metadata["preview"],
                },
            }
        )

    return results


def filter_by_distance(
    retrieved_chunks: list[dict],
    max_distance: float | None = None,
) -> list[dict]:
    """
    distance가 너무 큰 검색 결과를 제거한다.

    Chroma에서는 distance가 작을수록 더 가까운 결과다.
    max_distance=None이면 필터링하지 않는다.
    """

    if max_distance is None:
        return retrieved_chunks

    filtered = []

    for chunk in retrieved_chunks:
        if chunk["distance"] <= max_distance:
            filtered.append(chunk)

    return filtered


def build_context(retrieved_chunks: list[dict]) -> str:
    """
    검색된 chunk들을 LLM에게 넣을 context 문자열로 만든다.
    """

    context_blocks = []

    for index, chunk in enumerate(retrieved_chunks, start=1):
        metadata = chunk["metadata"]

        block = f"""
[Source {index}]
file_name: {metadata["file_name"]}
category: {metadata["category"]}
chunk_index: {metadata["chunk_index"]}
distance: {chunk["distance"]:.4f}

content:
{chunk["content"]}
""".strip()

        context_blocks.append(block)

    return "\n\n---\n\n".join(context_blocks)


def build_rag_prompt(query: str, context: str) -> str:
    """
    RAG용 prompt를 만든다.
    실제 LLM에게는 이 prompt를 전달하게 된다.
    """

    prompt = f"""
당신은 문서 기반 질의응답 assistant입니다.

규칙:
1. 반드시 제공된 Context 안의 정보만 사용하세요.
2. Context에 없는 내용은 추측하지 마세요.
3. 답변 마지막에는 사용한 Source 번호를 표시하세요.
4. 근거가 부족하면 "제공된 문서만으로는 답변하기 어렵습니다."라고 답하세요.

Context:
{context}

Question:
{query}

Answer:
""".strip()

    return prompt


def mock_llm_answer(query: str, retrieved_chunks: list[dict]) -> str:
    """
    실제 LLM 대신 사용하는 mock answer 함수.

    이 함수는 진짜 생성 모델이 아니다.
    검색된 chunk를 바탕으로 '근거 기반 답변처럼 보이는' 템플릿 답변을 만든다.
    """

    if not retrieved_chunks:
        return "제공된 문서만으로는 답변하기 어렵습니다."

    top_chunk = retrieved_chunks[0]
    top_metadata = top_chunk["metadata"]

    answer = f"""
제공된 문서 기준으로 보면, 질문 "{query}"에 대한 핵심 근거는 다음과 같습니다.

{top_chunk["content"][:500]}

요약하면, 이 질문은 주로 "{top_metadata["category"]}" 카테고리의 내용과 관련이 있습니다. 
가장 관련성이 높은 근거는 {top_metadata["file_name"]}의 chunk {top_metadata["chunk_index"]}입니다.

Sources:
- Source 1: {top_metadata["file_name"]} / chunk {top_metadata["chunk_index"]}
""".strip()

    return answer


def build_sources(retrieved_chunks: list[dict]) -> list[dict]:
    """
    답변에 표시할 source 목록을 만든다.
    """

    sources = []

    for index, chunk in enumerate(retrieved_chunks, start=1):
        metadata = chunk["metadata"]

        sources.append(
            {
                "source_number": index,
                "chunk_id": chunk["chunk_id"],
                "file_name": metadata["file_name"],
                "category": metadata["category"],
                "chunk_index": metadata["chunk_index"],
                "distance": chunk["distance"],
                "preview": metadata["preview"],
            }
        )

    return sources


def run_rag(
    query: str,
    n_results: int = 5,
    max_distance: float | None = None,
) -> dict:
    """
    Mini RAG 전체 파이프라인.
    """

    retrieved_chunks = retrieve_relevant_chunks(
        query=query,
        n_results=n_results,
    )

    filtered_chunks = filter_by_distance(
        retrieved_chunks=retrieved_chunks,
        max_distance=max_distance,
    )

    context = build_context(filtered_chunks)
    prompt = build_rag_prompt(query, context)
    answer = mock_llm_answer(query, filtered_chunks)
    sources = build_sources(filtered_chunks)

    result = {
        "query": query,
        "answer": answer,
        "sources": sources,
        "prompt": prompt,
        "retrieved_chunks": filtered_chunks,
    }

    return result


def save_result(result: dict, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    json_text = json.dumps(result, ensure_ascii=False, indent=2)
    path.write_text(json_text, encoding="utf-8")


def print_rag_result(result: dict) -> None:
    print("=" * 80)
    print("Mini RAG Result")
    print("=" * 80)

    print("\n[Question]")
    print(result["query"])

    print("\n[Answer]")
    print(result["answer"])

    print("\n[Sources]")
    for source in result["sources"]:
        print("-" * 80)
        print(f"Source {source['source_number']}")
        print(f"Chunk ID: {source['chunk_id']}")
        print(f"File: {source['file_name']}")
        print(f"Category: {source['category']}")
        print(f"Chunk Index: {source['chunk_index']}")
        print(f"Distance: {source['distance']:.4f}")
        print(f"Preview: {source['preview']}")

    print("\n[Prompt Preview]")
    print(result["prompt"][:1200])


def main():
    output_path = "phase0/lesson07/outputs/mini_rag_result.json"

    query = input("질문을 입력하세요: ").strip()

    if not query:
        print("질문이 비어 있습니다.")
        return

    result = run_rag(
        query=query,
        n_results=5,
        max_distance=None,
    )

    print_rag_result(result)
    save_result(result, output_path)

    print("\n결과 저장 완료")
    print(f"저장 위치: {output_path}")


if __name__ == "__main__":
    main()