from pathlib import Path
import json
from datetime import datetime

from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def infer_category(file_name: str, content: str) -> str:
    lower_text = f"{file_name} {content}".lower()

    if "rag" in lower_text or "retrieval" in lower_text:
        return "rag"

    if "fine" in lower_text or "tuning" in lower_text or "lora" in lower_text:
        return "fine_tuning"

    if "transformer" in lower_text or "attention" in lower_text:
        return "transformer"

    return "general"


def load_text_files(data_dir: str) -> list[dict]:
    directory = Path(data_dir)

    if not directory.exists():
        raise FileNotFoundError(f"폴더를 찾을 수 없습니다: {data_dir}")

    txt_files = sorted(directory.glob("*.txt"))
    documents = []

    for doc_index, txt_file in enumerate(txt_files, start=1):
        content = txt_file.read_text(encoding="utf-8")
        category = infer_category(txt_file.name, content)

        documents.append(
            {
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
        )

    return documents


def split_text_by_char(
    text: str,
    chunk_size: int = 300,
    chunk_overlap: int = 50,
) -> list[dict]:
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

            all_chunks.append(
                {
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
            )

    return all_chunks


def build_embedding_index(chunks: list[dict], model_name: str) -> list[dict]:
    print(f"Embedding model 로딩 중: {model_name}")
    model = SentenceTransformer(model_name)

    texts = [chunk["content"] for chunk in chunks]

    print(f"총 chunk 수: {len(texts)}")
    print("Embedding 생성 중...")

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    indexed_chunks = []

    for chunk, embedding in zip(chunks, embeddings):
        indexed_chunk = {
            "id": chunk["id"],
            "document_id": chunk["document_id"],
            "content": chunk["content"],
            "metadata": chunk["metadata"],
            "embedding": embedding.tolist(),
        }

        indexed_chunks.append(indexed_chunk)

    return indexed_chunks


def save_json(data: list[dict], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    json_text = json.dumps(data, ensure_ascii=False, indent=2)
    path.write_text(json_text, encoding="utf-8")


def main():
    data_dir = "phase0/lesson05/data"
    output_file = "phase0/lesson05/outputs/embedding_index.json"

    chunk_size = 300
    chunk_overlap = 50

    documents = load_text_files(data_dir)
    chunks = create_chunks(
        documents=documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    indexed_chunks = build_embedding_index(
        chunks=chunks,
        model_name=MODEL_NAME,
    )

    save_json(indexed_chunks, output_file)

    print("=" * 80)
    print("Embedding Index 생성 완료")
    print(f"문서 수: {len(documents)}")
    print(f"Chunk 수: {len(chunks)}")
    print(f"저장 위치: {output_file}")

    if indexed_chunks:
        first_embedding = indexed_chunks[0]["embedding"]
        print(f"Embedding 차원 수: {len(first_embedding)}")
        print(f"첫 번째 chunk ID: {indexed_chunks[0]['id']}")
        print(f"첫 번째 chunk 미리보기: {indexed_chunks[0]['metadata']['preview']}")


if __name__ == "__main__":
    main()