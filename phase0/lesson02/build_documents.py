from pathlib import Path
import json
from datetime import datetime


def infer_category(file_name: str, content: str) -> str:
    """
    파일명과 내용을 보고 간단한 카테고리를 추론한다.
    지금은 매우 단순한 규칙 기반이다.
    """

    lower_text = f"{file_name} {content}".lower()

    if "rag" in lower_text or "retrieval" in lower_text:
        return "rag"

    if "fine" in lower_text or "tuning" in lower_text or "lora" in lower_text:
        return "fine_tuning"

    if "transformer" in lower_text or "attention" in lower_text:
        return "transformer"

    return "general"


def load_text_file(file_path: str, doc_id: str) -> dict:
    """
    텍스트 파일 하나를 읽어서 RAG에서 쓰기 좋은 document 구조로 변환한다.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

    if not path.is_file():
        raise ValueError(f"파일이 아닙니다: {file_path}")

    content = path.read_text(encoding="utf-8")
    category = infer_category(path.name, content)

    document = {
        "id": doc_id,
        "content": content,
        "metadata": {
            "file_name": path.name,
            "file_path": str(path),
            "extension": path.suffix,
            "category": category,
            "source_type": "local_text_file",
            "text_length": len(content),
            "preview": content[:100],
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
    }

    return document


def load_documents_from_dir(data_dir: str) -> list[dict]:
    """
    폴더 안의 모든 .txt 파일을 읽어서 document list로 만든다.
    """

    directory = Path(data_dir)

    if not directory.exists():
        raise FileNotFoundError(f"폴더를 찾을 수 없습니다: {data_dir}")

    txt_files = sorted(directory.glob("*.txt"))

    documents = []

    for index, txt_file in enumerate(txt_files, start=1):
        doc_id = f"doc_{index:03d}"
        document = load_text_file(str(txt_file), doc_id)
        documents.append(document)

    return documents


def save_json(data: list[dict], output_path: str) -> None:
    """
    데이터를 JSON 파일로 저장한다.
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    json_text = json.dumps(data, ensure_ascii=False, indent=2)
    path.write_text(json_text, encoding="utf-8")


def main():
    data_dir = "phase0/lesson02/data"
    output_file = "phase0/lesson02/outputs/documents_with_metadata.json"

    documents = load_documents_from_dir(data_dir)
    save_json(documents, output_file)

    print("문서 빌드 완료")
    print(f"문서 수: {len(documents)}")
    print(f"저장 위치: {output_file}")

    for doc in documents:
        metadata = doc["metadata"]
        print("-" * 60)
        print(f"ID: {doc['id']}")
        print(f"파일명: {metadata['file_name']}")
        print(f"카테고리: {metadata['category']}")
        print(f"글자 수: {metadata['text_length']}")
        print(f"미리보기: {metadata['preview']}")


if __name__ == "__main__":
    main()