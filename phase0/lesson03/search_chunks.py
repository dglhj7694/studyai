from pathlib import Path
import json


def load_chunks(json_path: str) -> list[dict]:
    """
    chunks.json 파일에서 chunk 리스트를 불러온다.
    """

    path = Path(json_path)

    if not path.exists():
        raise FileNotFoundError(f"JSON 파일을 찾을 수 없습니다: {json_path}")

    json_text = path.read_text(encoding="utf-8")
    chunks = json.loads(json_text)

    return chunks


def count_keyword_occurrences(text: str, keyword: str) -> int:
    """
    텍스트 안에 키워드가 몇 번 등장하는지 센다.
    """

    return text.lower().count(keyword.lower())


def search_chunks(
    chunks: list[dict],
    query: str,
    min_score: int = 1,
    top_k: int = 5,
) -> list[dict]:
    """
    chunk 리스트에서 query가 등장하는 chunk를 검색한다.
    """

    results = []

    for chunk in chunks:
        content = chunk["content"]
        metadata = chunk["metadata"]

        content_count = count_keyword_occurrences(content, query)
        file_name_count = count_keyword_occurrences(metadata["file_name"], query)
        category_count = count_keyword_occurrences(metadata["category"], query)

        score = content_count + file_name_count * 2 + category_count * 3

        if score >= min_score:
            result = {
                "chunk_id": chunk["id"],
                "document_id": chunk["document_id"],
                "score": score,
                "file_name": metadata["file_name"],
                "category": metadata["category"],
                "chunk_index": metadata["chunk_index"],
                "start_char": metadata["start_char"],
                "end_char": metadata["end_char"],
                "preview": metadata["preview"],
                "content": content,
                "content_count": content_count,
                "file_name_count": file_name_count,
                "category_count": category_count,
            }

            results.append(result)

    results.sort(key=lambda x: x["score"], reverse=True)

    return results[:top_k]


def save_search_results(results: list[dict], output_path: str) -> None:
    """
    검색 결과를 JSON 파일로 저장한다.
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    json_text = json.dumps(results, ensure_ascii=False, indent=2)
    path.write_text(json_text, encoding="utf-8")


def print_results(query: str, results: list[dict]) -> None:
    """
    검색 결과 출력.
    """

    print("=" * 70)
    print(f"검색어: {query}")
    print(f"검색 결과 수: {len(results)}")
    print("=" * 70)

    if not results:
        print("검색 결과가 없습니다.")
        return

    for index, result in enumerate(results, start=1):
        print(f"\n[{index}] {result['file_name']} / chunk {result['chunk_index']}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Document ID: {result['document_id']}")
        print(f"점수: {result['score']}")
        print(f"카테고리: {result['category']}")
        print(f"문자 범위: {result['start_char']} ~ {result['end_char']}")
        print(f"미리보기: {result['preview']}")
        print("\n본문:")
        print(result["content"][:300])


def main():
    chunks_path = "phase0/lesson03/outputs/chunks.json"
    output_path = "phase0/lesson03/outputs/chunk_search_results.json"

    chunks = load_chunks(chunks_path)

    query = input("검색어를 입력하세요: ").strip()

    # 테스트용 고정 검색어를 쓰고 싶으면 위 줄을 주석 처리하고 아래 줄 사용
    # query = "벡터 데이터베이스"

    if not query:
        print("검색어가 비어 있습니다.")
        return

    results = search_chunks(
        chunks=chunks,
        query=query,
        min_score=1,
        top_k=5,
    )

    print_results(query, results)
    save_search_results(results, output_path)

    print("\n검색 결과 저장 완료")
    print(f"저장 위치: {output_path}")


if __name__ == "__main__":
    main()