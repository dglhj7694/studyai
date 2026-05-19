from pathlib import Path
from datetime import datetime

import chromadb
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "lesson06_rag_chunks"
CHROMA_PATH = "phase0/lesson06/chroma_db"


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
        doc_metadata = document["metadata"]

        raw_chunks = split_text_by_char(
            text=content,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for chunk_index, raw_chunk in enumerate(raw_chunks, start=1):
            chunk_id = f"{doc_id}_chunk_{chunk_index:03d}"

            metadata = {
                "document_id": doc_id,
                "chunk_index": chunk_index,
                "start_char": raw_chunk["start_char"],
                "end_char": raw_chunk["end_char"],
                "chunk_size": len(raw_chunk["text"]),
                "file_name": doc_metadata["file_name"],
                "file_path": doc_metadata["file_path"],
                "category": doc_metadata["category"],
                "source_type": doc_metadata["source_type"],
                "document_text_length": doc_metadata["text_length"],
                "preview": raw_chunk["text"][:100],
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }

            all_chunks.append(
                {
                    "id": chunk_id,
                    "content": raw_chunk["text"],
                    "metadata": metadata,
                }
            )

    return all_chunks


def reset_collection(client: chromadb.PersistentClient, collection_name: str):
    """
    실습에서는 매번 collection을 삭제하고 새로 만든다.
    같은 ID를 다시 add하면 Chroma가 기존 ID를 무시할 수 있기 때문이다.
    """

    try:
        client.delete_collection(name=collection_name)
        print(f"기존 collection 삭제 완료: {collection_name}")
    except Exception:
        print(f"삭제할 기존 collection이 없습니다: {collection_name}")

    collection = client.create_collection(
        name=collection_name,
        embedding_function=None,
        metadata={
            "description": "Phase 0 Lesson 06 Chroma vector store",
            "embedding_model": MODEL_NAME,
        },
    )

    return collection


def build_chroma_index(chunks: list[dict]) -> None:
    print(f"Embedding model 로딩 중: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    texts = [chunk["content"] for chunk in chunks]

    print(f"총 chunk 수: {len(texts)}")
    print("Embedding 생성 중...")

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).tolist()

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = reset_collection(client, COLLECTION_NAME)

    ids = [chunk["id"] for chunk in chunks]
    documents = [chunk["content"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

    print("Chroma collection에 데이터 저장 중...")

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    print("=" * 80)
    print("Chroma Index 생성 완료")
    print(f"Collection name: {COLLECTION_NAME}")
    print(f"저장 경로: {CHROMA_PATH}")
    print(f"저장된 chunk 수: {collection.count()}")


def main():
    data_dir = "phase0/lesson06/data"

    chunk_size = 300
    chunk_overlap = 50

    documents = load_text_files(data_dir)
    chunks = create_chunks(
        documents=documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    build_chroma_index(chunks)


if __name__ == "__main__":
    main()