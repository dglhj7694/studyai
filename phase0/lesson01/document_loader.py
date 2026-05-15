from pathlib import Path
import json


def load_text_file(file_path: str) -> dict:
    """
    텍스트 파일 하나를 읽어서 문서 정보로 변환한다.
    나중에 RAG의 Document Loader 기초가 되는 함수다.
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
        "content": text,
    }

    return document


def save_as_json(data: dict, output_path: str) -> None:
    """
    Python dict 데이터를 JSON 파일로 저장한다.
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    json_text = json.dumps(data, ensure_ascii=False, indent=2)
    path.write_text(json_text, encoding="utf-8")


def main():
    input_file = "phase0/lesson01/data/sample_note.txt"
    output_file = "phase0/lesson01/outputs/sample_note.json"

    document = load_text_file(input_file)
    save_as_json(document, output_file)

    print("문서 로딩 완료")
    print(f"파일명: {document['file_name']}")
    print(f"글자 수: {document['text_length']}")
    print(f"결과 저장 위치: {output_file}")


if __name__ == "__main__":
    main()