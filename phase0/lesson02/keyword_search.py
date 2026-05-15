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


def search_documents(documents: list[dict], query: str) -> list[dict]:
    """
    문서 리스트에서 query가 많이 등장하는 문서를 검색한다.
    """

    results = []

    for doc in documents:
        content = doc["content"]
        metadata = doc["metadata"]

        content_count = count_keyword_occurrences(content, query)
        file_name_count = count_keyword_occurrences(metadata["file_name"], query)
        category_count = count_keyword_occurrences(metadata["category"], query)

        score = content_count + file_name_count * 2 + category_count * 3

        if score > 0:
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

    documents = load_documents(json_path)

    query = input("검색어를 입력하세요: ").strip()

    if not query:
        print("검색어가 비어 있습니다.")
        return

    results = search_documents(documents, query)
    print_search_results(query, results)


if __name__ == "__main__":
    main()