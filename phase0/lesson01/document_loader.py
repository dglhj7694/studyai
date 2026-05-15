from pathlib import Path
import json


def load_text_file(file_path: str) -> dict:
    """
    텍스트 파일 하나를 읽어서 문서 정보로 변환한다.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

    if not path.is_file():
        raise ValueError(f"파일이 아닙니다: {file_path}")

    text = path.read_text(encoding="utf-8")

    document = {
        "file_name": path.name,
        "file_path": str(path),
        "extension": path.suffix,
        "text_length": len(text),
        "preview": text[:100],
        "content": text,
    }

    return document


def load_text_files_from_dir(data_dir: str) -> list[dict]:
    """
    특정 폴더 안에 있는 모든 .txt 파일을 읽어서 문서 리스트로 만든다.
    """

    directory = Path(data_dir)

    if not directory.exists():
        raise FileNotFoundError(f"폴더를 찾을 수 없습니다: {data_dir}")

    if not directory.is_dir():
        raise ValueError(f"폴더가 아닙니다: {data_dir}")

    txt_files = list(directory.glob("*.txt"))

    documents = []

    for txt_file in txt_files:
        document = load_text_file(str(txt_file))
        documents.append(document)

    return documents


def save_as_json(data: list[dict], output_path: str) -> None:
    """
    Python list/dict 데이터를 JSON 파일로 저장한다.
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    json_text = json.dumps(data, ensure_ascii=False, indent=2)
    path.write_text(json_text, encoding="utf-8")


def main():
    data_dir = "phase0/lesson01/data"
    output_file = "phase0/lesson01/outputs/documents.json"

    documents = load_text_files_from_dir(data_dir)
    save_as_json(documents, output_file)

    print("여러 문서 로딩 완료")
    print(f"읽은 문서 수: {len(documents)}")
    print(f"결과 저장 위치: {output_file}")

    for document in documents:
        print("-" * 50)
        print(f"파일명: {document['file_name']}")
        print(f"글자 수: {document['text_length']}")
        print(f"미리보기: {document['preview']}")


if __name__ == "__main__":
    main()