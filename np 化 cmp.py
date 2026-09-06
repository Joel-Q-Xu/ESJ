import numpy as np
import timeit

binary_dig = "133" * 3  # 自己改长度

cases = {
    "fromiter_gen": """
np.fromiter(
    (1 if c == '1' else 0 for c in binary_dig),
    dtype=np.int8,
    count=len(binary_dig),
)
""",
    "fromiter_ord": """
np.fromiter(binary_dig, dtype=np.int8, count=len(binary_dig)) - ord('0')
""",
    "array_list": """
np.array([int(bit) for bit in binary_dig], dtype=np.int8)
""",
}

for name, stmt in cases.items():
    t = min(timeit.repeat(stmt, number=50000, repeat=5, globals=globals()))
    print(f"{name:<12} {t:.6f}s")