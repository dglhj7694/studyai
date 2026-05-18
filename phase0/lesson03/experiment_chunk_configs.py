from chunk_documents import load_text_files, create_chunks


def run_experiment():
    data_dir = "phase0/lesson03/data"

    configs = [
        {"chunk_size": 150, "chunk_overlap": 30},
        {"chunk_size": 300, "chunk_overlap": 50},
        {"chunk_size": 500, "chunk_overlap": 100},
    ]

    documents = load_text_files(data_dir)

    print("=" * 80)
    print("Chunk 설정별 실험 결과")
    print("=" * 80)

    for config in configs:
        chunk_size = config["chunk_size"]
        chunk_overlap = config["chunk_overlap"]

        chunks = create_chunks(
            documents=documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        print("-" * 80)
        print(f"chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
        print(f"총 문서 수: {len(documents)}")
        print(f"총 chunk 수: {len(chunks)}")

        avg_chunk_size = sum(
            chunk["metadata"]["chunk_size"] for chunk in chunks
        ) / len(chunks)

        print(f"평균 chunk 길이: {avg_chunk_size:.1f}")

        print("\n샘플 chunk:")
        sample_chunk = chunks[0]
        print(f"Chunk ID: {sample_chunk['id']}")
        print(f"파일명: {sample_chunk['metadata']['file_name']}")
        print(f"미리보기: {sample_chunk['metadata']['preview']}")


if __name__ == "__main__":
    run_experiment()