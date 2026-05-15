from pathlib import Path
import json


def load_documents(json_path: str) -> list[dict]:
    """
    JSON 파일에서 문서 리스트를 불러온다.
    """

    path = Path(json_path)

    if not path.exists():
        raise FileNotFoundError(f"JSON 파일을 찾을 수 없습니다: {json_path}")

    json_text = path.read_text(encoding="utf-8")
    documents = json.loads(json_text)

    return documents


def count_keyword_occurrences(text: str, keyword: str) -> int:
    """
    텍스트 안에 키워드가 몇 번 등장하는지 센다.
    대소문자 구분을 없애기 위해 lower()를 사용한다.
    """

    text_lower = text.lower()
    keyword_lower = keyword.lower()

    return text_lower.count(keyword_lower)


def search_documents(
    documents: list[dict],
    query: str,
    min_score: int = 1,
) -> list[dict]:
    """
    문서 리스트에서 query가 많이 등장하는 문서를 검색한다.

    min_score:
        검색 결과에 포함되기 위한 최소 점수
    """

    results = []

    for doc in documents:
        content = doc["content"]
        metadata = doc["metadata"]

        content_count = count_keyword_occurrences(content, query)
        file_name_count = count_keyword_occurrences(metadata["file_name"], query)
        category_count = count_keyword_occurrences(metadata["category"], query)

        score = content_count + file_name_count * 2 + category_count * 3

        if score >= min_score:
            result = {
                "id": doc["id"],
                "score": score,
                "file_name": metadata["file_name"],
                "category": metadata["category"],
                "preview": metadata["preview"],
                "content_count": content_count,
                "file_name_count": file_name_count,
                "category_count": category_count,
            }

            results.append(result)

    results.sort(key=lambda x: x["score"], reverse=True)

    return results


def save_search_results(results: list[dict], output_path: str) -> None:
    """
    검색 결과를 JSON 파일로 저장한다.
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    json_text = json.dumps(results, ensure_ascii=False, indent=2)
    path.write_text(json_text, encoding="utf-8")


def print_search_results(query: str, results: list[dict]) -> None:
    """
    검색 결과를 보기 좋게 출력한다.
    """

    print("=" * 70)
    print(f"검색어: {query}")
    print(f"검색 결과 수: {len(results)}")
    print("=" * 70)

    if not results:
        print("검색 결과가 없습니다.")
        return

    for index, result in enumerate(results, start=1):
        print(f"\n[{index}] {result['file_name']}")
        print(f"ID: {result['id']}")
        print(f"점수: {result['score']}")
        print(f"카테고리: {result['category']}")
        print(f"본문 등장 횟수: {result['content_count']}")
        print(f"파일명 등장 횟수: {result['file_name_count']}")
        print(f"카테고리 등장 횟수: {result['category_count']}")
        print(f"미리보기: {result['preview']}")


def main():
    json_path = "phase0/lesson02/outputs/documents_with_metadata.json"
    output_path = "phase0/lesson02/outputs/search_results.json"

    documents = load_documents(json_path)

    # 방법 1. 직접 입력받기
    query = input("검색어를 입력하세요: ").strip()

    # 방법 2. 테스트용 고정 검색어를 쓰고 싶으면 위 줄을 주석 처리하고 아래 줄 사용
    # query = "RAG"

    if not query:
        print("검색어가 비어 있습니다.")
        return

    min_score = 1

    results = search_documents(
        documents=documents,
        query=query,
        min_score=min_score,
    )

    print_search_results(query, results)
    save_search_results(results, output_path)

    print("\n검색 결과 JSON 저장 완료")
    print(f"저장 위치: {output_path}")


if __name__ == "__main__":
    main()