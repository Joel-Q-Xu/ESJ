import hashlib
import timeit
import mmh3
import fnvhash

def mixed_hash_bits(input_string: str, l: int) -> str:
    data = input_string.encode("utf-8")

    if l == 32:
        value = mmh3.hash(input_string, signed=False)
        return format(value, "032b")
    elif l == 64:
        value = fnvhash.fnv1a_64(data)
        return format(value, "064b")
    elif l == 128:
        value = mmh3.hash128(input_string, seed=0, signed=False)
        return format(value, "0128b")
    elif l == 256:
        v1 = mmh3.hash128(input_string, seed=0, signed=False)
        v2 = mmh3.hash128(input_string, seed=1, signed=False)
        return f"{v1:0128b}{v2:0128b}"
    else:
        raise ValueError

def shake_hash_bits(input_string: str, l: int) -> str:
    digest = hashlib.shake_256(input_string.encode("utf-8")).digest(l // 8)
    value = int.from_bytes(digest, "big")
    return format(value, f"0{l}b")

setup = """
from __main__ import mixed_hash_bits, shake_hash_bits
s = "这是一个测试字符串，用来比较不同哈希函数的性能。" * 4
"""

cases = {
    "mixed_32" : 'mixed_hash_bits(s, 32)',
    "shake_32" : 'shake_hash_bits(s, 32)',
    "mixed_64" : 'mixed_hash_bits(s, 64)',
    "shake_64" : 'shake_hash_bits(s, 64)',
    "mixed_128": 'mixed_hash_bits(s, 128)',
    "shake_128": 'shake_hash_bits(s, 128)',
    "mixed_256": 'mixed_hash_bits(s, 256)',
    "shake_256": 'shake_hash_bits(s, 256)',
}

number = 100000
repeat = 5

for name, stmt in cases.items():
    best = min(timeit.repeat(stmt=stmt, setup=setup, number=number, repeat=repeat))
    print(f"{name:<10} {best:.6f}s")


setup = """
import mmh3, hashlib

s = "这是一个测试字符串，用来比较不同哈希函数的性能。" * 4

def mmh3_64_from_128(s):
    v128 = mmh3.hash128(s, seed=0, signed=False)
    return v128 & ((1 << 64) - 1)

def shake_64(s):
    return hashlib.shake_256(s.encode("utf-8")).digest(8)
"""

print("mmh3_64_from_128:", min(timeit.repeat("mmh3_64_from_128(s)", setup=setup, number=100000, repeat=5)))
print("shake_64        :", min(timeit.repeat("shake_64(s)", setup=setup, number=100000, repeat=5)))