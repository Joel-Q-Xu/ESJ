import hashlib
import random
import time
import statistics
import numpy as np
import mmh3

p = 3015000013
key_t = "example-key"
bit_lengths = [32, 64, 128, 256]

def f_random_list(key_t: str, bit_length: int, p: int) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(key_t.encode("utf-8")).digest(), "big")
    rng = random.Random(seed)
    return np.array([rng.randrange(1, p) for _ in range(bit_length)], dtype=np.int64)

def f_random_fromiter(key_t: str, bit_length: int, p: int) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(key_t.encode("utf-8")).digest(), "big")
    rng = random.Random(seed)
    return np.fromiter(
        (rng.randrange(1, p) for _ in range(bit_length)),
        dtype=np.int64,
        count=bit_length
    )

def f_mmh3(key_t: str, bit_length: int, p: int) -> np.ndarray:
    return np.array(
        [(mmh3.hash(f"{key_t}:{i}", signed=False) % (p - 1)) + 1 for i in range(bit_length)],
        dtype=np.int64
    )

def f_numpy(key_t: str, bit_length: int, p: int) -> np.ndarray:
    digest = hashlib.sha256(key_t.encode("utf-8")).digest()
    seed_words = np.frombuffer(digest, dtype=np.uint32)
    rng = np.random.default_rng(np.random.SeedSequence(seed_words))
    return rng.integers(1, p, size=bit_length, dtype=np.int64)

funcs = [
    ("random+list", f_random_list),
    ("random+fromiter", f_random_fromiter),
    ("mmh3", f_mmh3),
    ("numpy.default_rng", f_numpy),
]

loops_map = {32: 10, 64: 100, 128: 10, 4: 100}
rounds = 7

for bl in bit_lengths:
    print(f"\nbit_length = {bl}")
    loops = loops_map[bl]
    for name, func in funcs:
        samples = []
        for _ in range(rounds):
            t0 = time.perf_counter()
            for _ in range(loops):
                func(key_t, bl, p)
            samples.append((time.perf_counter() - t0) / loops * 1e6)
        print(f"{name:18s}: median = {statistics.median(samples):8.2f} us")