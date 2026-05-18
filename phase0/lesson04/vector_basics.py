import math


def dot_product(vector_a: list[float], vector_b: list[float]) -> float:
    """
    두 벡터의 내적을 계산한다.
    """

    if len(vector_a) != len(vector_b):
        raise ValueError("두 벡터의 길이가 같아야 합니다.")

    total = 0.0

    for a, b in zip(vector_a, vector_b):
        total += a * b

    return total


def vector_norm(vector: list[float]) -> float:
    """
    벡터의 길이를 계산한다.
    """

    squared_sum = 0.0

    for value in vector:
        squared_sum += value ** 2

    return math.sqrt(squared_sum)


def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    """
    두 벡터의 코사인 유사도를 계산한다.
    """

    dot = dot_product(vector_a, vector_b)
    norm_a = vector_norm(vector_a)
    norm_b = vector_norm(vector_b)

    if norm_a == 0 or norm_b == 0:
        raise ValueError("0 벡터는 코사인 유사도를 계산할 수 없습니다.")

    return dot / (norm_a * norm_b)


def main():
    vector_a = [1, 0]
    vector_b = [0, 1]
    vector_c = [1, 1]
    vector_d = [2, 2]

    print("=" * 70)
    print("벡터 기본 연산")
    print("=" * 70)

    print(f"A: {vector_a}")
    print(f"B: {vector_b}")
    print(f"C: {vector_c}")
    print(f"D: {vector_d}")

    print("\n내적:")
    print(f"A · B = {dot_product(vector_a, vector_b)}")
    print(f"A · C = {dot_product(vector_a, vector_c)}")
    print(f"C · D = {dot_product(vector_c, vector_d)}")

    print("\n벡터 길이:")
    print(f"||A|| = {vector_norm(vector_a):.4f}")
    print(f"||B|| = {vector_norm(vector_b):.4f}")
    print(f"||C|| = {vector_norm(vector_c):.4f}")
    print(f"||D|| = {vector_norm(vector_d):.4f}")

    print("\n코사인 유사도:")
    print(f"cos(A, B) = {cosine_similarity(vector_a, vector_b):.4f}")
    print(f"cos(A, C) = {cosine_similarity(vector_a, vector_c):.4f}")
    print(f"cos(C, D) = {cosine_similarity(vector_c, vector_d):.4f}")


if __name__ == "__main__":
    main()