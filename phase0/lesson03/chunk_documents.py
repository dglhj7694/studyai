from pathlib import Path
import json
from datetime import datetime


def infer_category(file_name: str, content: str) -> str:
    """
    파일명과 내용을 보고 간단한 카테고리를 추론한다.
    """

    lower_text = f"{file_name} {content}".lower()

    if "rag" in lower_text or "retrieval" in lower_text:
        return "rag"

    if "fine" in lower_text or "tuning" in lower_text or "lora" in lower_text:
        return "fine_tuning"

    if "transformer" in lower_text or "attention" in lower_text:
        return "transformer"

    return "general"


def load_text_files(data_dir: str) -> list[dict]:
    """
    폴더 안의 모든 txt 파일을 document 구조로 로딩한다.
    """

    directory = Path(data_dir)

    if not directory.exists():
        raise FileNotFoundError(f"폴더를 찾을 수 없습니다: {data_dir}")

    txt_files = sorted(directory.glob("*.txt"))

    documents = []

    for doc_index, txt_file in enumerate(txt_files, start=1):
        content = txt_file.read_text(encoding="utf-8")
        category = infer_category(txt_file.name, content)

        document = {
            "id": f"doc_{doc_index:03d}",
            "content": content,
            "metadata": {
                "file_name": txt_file.name,
                "file_path": str(txt_file),
                "extension": txt_file.suffix,
                "category": category,
                "source_type": "local_text_file",
                "text_length": len(content),
                "preview": content[:100],
                "created_at": datetime.now().isoformat(timespec="seconds"),
            },
        }

        documents.append(document)

    return documents


def split_text_by_char(
    text: str,
    chunk_size: int = 300,
    chunk_overlap: int = 50,
) -> list[dict]:
    """
    텍스트를 글자 수 기준으로 chunk로 나눈다.

    chunk_size:
        chunk 하나의 최대 글자 수

    chunk_overlap:
        다음 chunk와 겹칠 글자 수
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size는 0보다 커야 합니다.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap은 0 이상이어야 합니다.")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap은 chunk_size보다 작아야 합니다.")

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append(
                {
                    "text": chunk_text,
                    "start_char": start,
                    "end_char": min(end, text_length),
                }
            )

        start = start + chunk_size - chunk_overlap

    return chunks


def create_chunks(
    documents: list[dict],
    chunk_size: int = 300,
    chunk_overlap: int = 50,
) -> list[dict]:
    """
    document 리스트를 chunk 리스트로 변환한다.
    """

    all_chunks = []

    for document in documents:
        doc_id = document["id"]
        content = document["content"]
        metadata = document["metadata"]

        raw_chunks = split_text_by_char(
            text=content,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for chunk_index, raw_chunk in enumerate(raw_chunks, start=1):
            chunk_id = f"{doc_id}_chunk_{chunk_index:03d}"

            chunk = {
                "id": chunk_id,
                "document_id": doc_id,
                "content": raw_chunk["text"],
                "metadata": {
                    "chunk_index": chunk_index,
                    "start_char": raw_chunk["start_char"],
                    "end_char": raw_chunk["end_char"],
                    "chunk_size": len(raw_chunk["text"]),
                    "file_name": metadata["file_name"],
                    "file_path": metadata["file_path"],
                    "category": metadata["category"],
                    "source_type": metadata["source_type"],
                    "document_text_length": metadata["text_length"],
                    "preview": raw_chunk["text"][:100],
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                },
            }

            all_chunks.append(chunk)

    return all_chunks


def save_json(data: list[dict], output_path: str) -> None:
    """
    데이터를 JSON 파일로 저장한다.
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    json_text = json.dumps(data, ensure_ascii=False, indent=2)
    path.write_text(json_text, encoding="utf-8")


def print_chunk_summary(chunks: list[dict]) -> None:
    """
    chunk 생성 결과를 요약 출력한다.
    """

    print("=" * 70)
    print("Chunk 생성 결과")
    print("=" * 70)
    print(f"총 chunk 수: {len(chunks)}")

    for chunk in chunks:
        metadata = chunk["metadata"]

        print("-" * 70)
        print(f"Chunk ID: {chunk['id']}")
        print(f"원본 문서 ID: {chunk['document_id']}")
        print(f"파일명: {metadata['file_name']}")
        print(f"카테고리: {metadata['category']}")
        print(f"Chunk 번호: {metadata['chunk_index']}")
        print(f"문자 범위: {metadata['start_char']} ~ {metadata['end_char']}")
        print(f"Chunk 길이: {metadata['chunk_size']}")
        print(f"미리보기: {metadata['preview']}")


def main():
    data_dir = "phase0/lesson03/data"
    output_file = "phase0/lesson03/outputs/chunks.json"

    chunk_size = 300
    chunk_overlap = 50

    documents = load_text_files(data_dir)

    chunks = create_chunks(
        documents=documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    save_json(chunks, output_file)
    print_chunk_summary(chunks)

    print("\n저장 완료")
    print(f"저장 위치: {output_file}")


if __name__ == "__main__":
    main()