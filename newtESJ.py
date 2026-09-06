import mmh3
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import csv
import sys, os, math
import time
from mpmath import mp
from numba import njit
from numba import prange,set_num_threads
sys.path.insert(0, os.path.abspath('charm'))
sys.path.insert(1, os.path.abspath('../charm'))
import random
from sympy import Matrix, GramSchmidt
from scipy.linalg import null_space
from scipy.linalg import lapack
import secrets
MAX_BASIS_TO_SUM = 32
AUX_DIM = 2
DOT_CHUNK = 1024
CALC_DTYPE = np.int64


def _add_scaled_kron_mod(
    dst: np.ndarray,
    basis_vec: np.ndarray,
    small_vec: np.ndarray,
    scale: int,
    mod: int,
) -> None:
    """
    dst += scale * kron(basis_vec, small_vec) (mod mod)
    不显式构造 np.kron 的大临时数组
    """
    block_len = small_vec.shape[0]
    scaled_basis = (basis_vec * scale) % mod

    for block_idx, coeff in enumerate(scaled_basis):
        if coeff == 0:
            continue
        start = block_idx * block_len
        end = start + block_len
        dst[start:end] = (dst[start:end] + coeff * small_vec) % mod


def _dot_mod_chunked(a: np.ndarray, b: np.ndarray, mod: int) -> int:
    """
    分块模点积，避免 object，也减少一次性大数组求和的压力
    """
    acc = 0
    n = a.shape[0]
    for start in range(0, n, DOT_CHUNK):
        end = min(start + DOT_CHUNK, n)
        chunk = (a[start:end] * b[start:end]) % mod
        acc = (acc + int(np.sum(chunk, dtype=np.int64))) % mod
    return acc

#-------------------VenGen------------------
def mod_inv(x: int, mod: int) -> int:
    """x 在 F_mod 上的乘法逆元。要求 mod 为素数且 x != 0。"""
    x %= mod
    if x == 0:
        raise ZeroDivisionError("0 在有限域上没有逆元")
    return pow(x, mod - 2, mod)




def dot_mod(a: List[int], b: List[int], mod: int) -> int:
    """
    标准内积（mod p）
    """
    if len(a) != len(b):
        raise ValueError("向量维度不一致")

    acc = 0
    for x, y in zip(a, b):
        acc = (acc + (x % mod) * (y % mod)) % mod
    return acc

def legendre_symbol(a: int, mod: int) -> int:
    a %= mod
    if a == 0:
        return 0
    t = pow(a, (mod - 1) // 2, mod)
    return -1 if t == mod - 1 else t

def is_nonzero_square(a: int, mod: int) -> bool:
    a %= mod
    return a != 0 and legendre_symbol(a, mod) == 1


def mod_sqrt(a: int, mod: int) -> int:
    a %= mod
    if a == 0:
        return 0
    if legendre_symbol(a, mod) != 1:
        raise ValueError("not a square")

    if mod % 4 == 3:
        return pow(a, (mod + 1) // 4, mod)

    q = mod - 1
    s = 0
    while q % 2 == 0:
        s += 1
        q //= 2

    z = 2
    while legendre_symbol(z, mod) != -1:
        z += 1

    m = s
    c = pow(z, q, mod)
    t = pow(a, q, mod)
    r = pow(a, (q + 1) // 2, mod)

    while t != 1:
        i = 1
        t2i = (t * t) % mod
        while i < m:
            if t2i == 1:
                break
            t2i = (t2i * t2i) % mod
            i += 1
        b = pow(c, 1 << (m - i - 1), mod)
        r = (r * b) % mod
        t = (t * b * b) % mod
        c = (b * b) % mod
        m = i

    return r

def orthogonal(
    l: int,
    p: int,
) -> List[List[int]]:
    """
    在 F_p^l 上生成一组真正的 orthonormal basis:
        <O[i], O[j]> = delta_{ij} mod p

    返回：
        O: 向量列表，长度为 l，每个向量也是长度为 l 的 list[int]

    说明：
    - 这里使用有限域版 Gram-Schmidt。
    - 每一步都要求 <u_i, u_i> 是非零平方元，才能归一化成长度 1。
    - 若随机重试次数过多，则抛出 RuntimeError。
    """

    O: List[List[int]] = []

    for i in range(l):
        while True:
            u = [random.randrange(p) for _ in range(l)]

            for j in range(i):
                oj = O[j]
                coeff = dot_mod(u, oj, p)
                if coeff:
                    for k in range(l):
                        u[k] = (u[k] - coeff * oj[k]) % p

            norm_sq = dot_mod(u, u, p)
            if not is_nonzero_square(norm_sq, p):
                continue

            scale = mod_sqrt(mod_inv(norm_sq, p), p)
            for k in range(l):
                u[k] = (u[k] * scale) % p

            if dot_mod(u, u, p) != 1:
                raise RuntimeError("normalization failed")

            O.append(u)
            break

    return O


def check_orthonormal(O: List[List[int]], mod: int) -> bool:
    """
    检查 O 是否为 orthonormal basis
    """
    l = len(O)
    for i in range(l):
        for j in range(l):
            val = dot_mod(O[i], O[j], mod)
            expected = 1 if i == j else 0
            if val != expected:
                return False
    return True

#---------------求Ax=0---------------------
def nullspace_modp_fast(rows: List[List[int]], mod: int) -> List[List[int]]:
    """
    求 A x = 0 (mod p) 的零空间基。

    rows:
        按“行”给出的矩阵 A。
    返回:
        零空间的一组基向量，每个向量长度 = A 的列数。
    """
    if len(rows) == 0:
        return []

    A = [[x % mod for x in row] for row in rows]
    m = len(A)
    n = len(A[0])

    pivot_cols: List[int] = []
    r = 0

    for c in range(n):
        pivot = -1
        for i in range(r, m):
            if A[i][c] != 0:
                pivot = i
                break
        if pivot == -1:
            continue

        if pivot != r:
            A[r], A[pivot] = A[pivot], A[r]

        inv_pivot = mod_inv(A[r][c], mod)
        for j in range(c, n):
            A[r][j] = (A[r][j] * inv_pivot) % mod

        for i in range(m):
            if i == r:
                continue
            factor = A[i][c]
            if factor != 0:
                for j in range(c, n):
                    A[i][j] = (A[i][j] - factor * A[r][j]) % mod

        pivot_cols.append(c)
        r += 1
        if r == m:
            break

    pivot_set = set(pivot_cols)
    basis: List[List[int]] = []

    for free in range(n):
        if free in pivot_set:
            continue

        x = [0] * n
        x[free] = 1
        for row_idx, pivot_col in enumerate(pivot_cols):
            x[pivot_col] = (-A[row_idx][free]) % mod
        basis.append(x)

    return basis


def choose_query_vector(candidate_values: Sequence[str], bit_length: int, mod: int,key) -> np.ndarray:
    """
    先求零空间基 basis，然后从 basis 里取最多 32 个基向量做线性组合：
    - basis >= 32：随机无放回抽 32 个
    - basis < 32 ：有多少取多少

    这是按你最新要求写的：
    “选 32 个 base 里的解加起来做最终解；少于 32 个就全加。”
    """
    rows = [hashlength_bits(key + v, bit_length).tolist() for v in candidate_values]
    basis = nullspace_modp_fast(rows, mod)
    if not basis:
        raise ValueError("给定约束下零空间为空，无法构造查询向量")

    basis_arr = np.asarray(basis, dtype=np.int64)

    take = min(MAX_BASIS_TO_SUM, len(basis_arr))
    indices = list(range(len(basis_arr))) if len(basis_arr) <= MAX_BASIS_TO_SUM else random.sample(range(len(basis_arr)), take)

    coeffs = np.array([random.randrange(1, mod) for _ in indices], dtype=np.int64)
    picked = basis_arr[indices]
    x = np.sum((picked * coeffs[:, None]) % mod, axis=0, dtype=np.int64) % mod

    if np.all(x == 0):
        x = basis_arr[0].copy()
    return x
# ============================================================
# 2. 哈希：
#    - F(value, l1)  -> l1 bit 向量（你原来的 hashlength 用法）
#    - H(value)      -> Z_p^* 中整数（你图里真的在用的 H(v^r_tau) ）
# ============================================================

def hashlength_bits(value: str, bit_length: int) -> np.ndarray:
    """
    输出截断后的 0/1 向量，长度支持 20/40/80/160 bit。
    做法：先生成足够长的哈希比特串，再取前 bit_length 位。
    位序保持不变：左边高位、右边低位。
    """
    if bit_length not in (20, 40, 80, 160):
        raise ValueError("bit_length 仅支持 20/40/80/160")

    if bit_length <= 32:
        h = mmh3.hash(value, signed=False)
        raw = h.to_bytes(4, byteorder="big", signed=False)   # 32 bit
    elif bit_length <= 64:
        h128 = mmh3.hash128(value, seed=0, signed=False)
        h = h128 & ((1 << 64) - 1)
        raw = h.to_bytes(8, byteorder="big", signed=False)   # 64 bit
    elif bit_length <= 128:
        h = mmh3.hash128(value, seed=0, signed=False)
        raw = h.to_bytes(16, byteorder="big", signed=False)  # 128 bit
    else:  # bit_length <= 160
        h1 = mmh3.hash128(value, seed=0, signed=False)
        h2 = mmh3.hash128(value, seed=1, signed=False)
        raw = h1.to_bytes(16, byteorder="big", signed=False) + h2.to_bytes(16, byteorder="big", signed=False)  # 256 bit

    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8)).astype(np.int64)
    return bits[:bit_length]

def H_modp_nonzero(s: str, mod: int) -> int:
    return 1 + (mmh3.hash(s) % (mod - 1))

def make_q_t(aux_basis_t, mod: int) -> np.ndarray:
    """
    构造 q_t，使得 <aux_basis_t, q_t> = 1 mod p
    假设 mod 是素数，且 aux_basis_t 全部来自 1..mod-1
    """
    a = np.asarray(aux_basis_t, dtype=np.int64) % mod
    n = len(a)
    if n == 0:
        raise ValueError("aux_basis_t is empty")

    if n == 1:
        return np.array([pow(int(a[0]), -1, mod)], dtype=np.int64)

    inv_last = pow(int(a[-1]), -1, mod)

    q = np.array([random.randrange(1, mod) for _ in range(n - 1)], dtype=np.int64)
    acc = 0
    for ai, qi in zip(a[:-1], q):
        acc = (acc + int(ai) * int(qi)) % mod

    q_last = ((1 - acc) * inv_last) % mod
    return np.concatenate((q, np.array([q_last], dtype=np.int64)))


def make_q_d(aux_basis_d, mod: int) -> np.ndarray:
    """
    构造非全零 q_d，使得 <aux_basis_d, q_d> = 0 mod p
    假设 mod 是素数，且 aux_basis_d 全部来自 1..mod-1
    """
    a = np.asarray(aux_basis_d, dtype=np.int64) % mod
    n = len(a)
    if n == 0:
        raise ValueError("aux_basis_d is empty")

    if n == 1:
        raise ValueError("AUX_DIM must be >= 2, otherwise nonzero q_d may not exist")

    inv_last = pow(int(a[-1]), -1, mod)

    q = np.array([random.randrange(1, mod) for _ in range(n - 1)], dtype=np.int64)
    acc = 0
    for ai, qi in zip(a[:-1], q):
        acc = (acc + int(ai) * int(qi)) % mod

    q_last = (-acc * inv_last) % mod
    return np.concatenate((q, np.array([q_last], dtype=np.int64)))

def read_table(path: str, delimiter: str = "|") -> List[Dict[str, str]]:
    """读取 .tbl，并去掉 DictReader 因行尾分隔符产生的 None 列。"""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter, skipinitialspace=False)
        rows: List[Dict[str, str]] = []
        for row in reader:
            clean_row = {k: str(v) for k, v in row.items() if k is not None}
            rows.append(clean_row)
    return rows


def decrypt(token_vec: np.ndarray, cipher_vec: np.ndarray, mod: int) -> int:
    token_vec = np.asarray(token_vec, dtype=np.int64)
    cipher_vec = np.asarray(cipher_vec, dtype=np.int64)
    return int(_dot_mod_chunked(token_vec, cipher_vec, mod))


def Setup(table,join_attrs,mod: int,):
    if not table:
        raise ValueError("Setup 需要非空表")

    encoded_attrs = list(table[0].keys())

    total_blocks = len(join_attrs)+len(encoded_attrs)

    main_basis = np.asarray(orthogonal(total_blocks, mod), dtype=np.int64)
    return encoded_attrs,main_basis

def Encrypt(
    table,
    encoded_attrs,
    join_attrs,
    pk_attr: str,
    main_basis,
    aux_basis_t,
    aux_basis_d,
    bit_length,
    mod,
    key,
):
    block_len = bit_length + AUX_DIM
    full_len = len(main_basis[0]) * block_len

    aux_t = np.asarray(aux_basis_t,  dtype=np.int64)
    aux_d = np.asarray(aux_basis_d,  dtype=np.int64)
    s_vec = np.empty(block_len, dtype=np.int64)
    s_vec[:bit_length] = aux_t
    s_vec[bit_length:] = aux_d

    offset = len(encoded_attrs)

    encrypted_table: List[Dict[str, object]] = []
    attr_u = np.empty(block_len, dtype=np.int64)
    attr_u[bit_length:] = aux_d

    for row in table:
        c_vec = np.zeros(full_len, dtype=np.int64)
        # join 属性块：对应 H(v_tau^r) * (s ⊗ o*_tau) 的“o*_tau 这一块系数向量”
        # for attr_name in join_attrs:
        #     h_value = H_modp_nonzero(row[attr_name], mod)
        #     slot_idx=offset+join_attrs.index(attr_name)
        #     c_vec = (c_vec + h_value * np.kron(main_basis[slot_idx], coeff_vec)) % mod
        for join_idx, join_attr in enumerate(join_attrs):
            h_value = H_modp_nonzero(row[join_attr], mod)
            _add_scaled_kron_mod(c_vec, main_basis[offset + join_idx], s_vec, h_value, mod)


        # 普通属性块：对应 [F(value) || d0 || d1]
        # for attr_name in encoded_attrs:
        for attr_idx, attr_name in enumerate(encoded_attrs):
            gamma = random.randrange(1, mod)
            attr_u[:bit_length] = hashlength_bits(key + row[attr_name], bit_length)
            _add_scaled_kron_mod(c_vec, main_basis[attr_idx], attr_u, gamma, mod)

        encrypted_table.append({"pk": row[pk_attr], "c": c_vec})

    return encrypted_table


def TokenGen(
    join_attrs,
    encoded_attrs,
    filter_query,
    main_basis,
    aux_basis_t,
    aux_basis_d,
    bit_length,
    mod,
    shared_join_scalar: int,
    q_join_attr,
    key,
):
    block_len = bit_length + AUX_DIM
    full_len = len(main_basis[0]) * block_len
    offset = len(encoded_attrs)

    q_t = make_q_t(aux_basis_t, mod)
    q_d = make_q_d(aux_basis_d, mod)
    e_vec = np.empty(block_len, dtype=np.int64)
    e_vec[:bit_length] = q_t
    e_vec[bit_length:] = q_d

    y_vec = np.zeros(block_len, dtype=np.int64)
    y_vec[bit_length:] = q_d


    tk_vec = np.zeros(full_len, dtype=np.int64)

    # 连接属性
    join_attr_to_idx = {name: i for i, name in enumerate(join_attrs)}
    encoded_attr_to_idx = {name: i for i, name in enumerate(encoded_attrs)}
    q_join_idx = join_attr_to_idx[q_join_attr]
    _add_scaled_kron_mod(tk_vec, main_basis[offset + q_join_idx], e_vec, shared_join_scalar, mod)

    for i in range(len(join_attrs)):
        if i == q_join_idx:
            continue
        alpha = random.randrange(1, mod)
        _add_scaled_kron_mod(tk_vec, main_basis[offset + i], y_vec, alpha, mod)

    used_attr = np.zeros(len(encoded_attrs), dtype=np.uint8)

    xi_vec = np.empty(block_len, dtype=np.int64)
    xi_vec[bit_length:] = q_d

    if filter_query is not None:
        for q_att_s, att_name in enumerate(filter_query[0]):
            beta = random.randrange(1, mod)
            candidate_values = filter_query[1][q_att_s]
            x_vec = choose_query_vector(candidate_values, bit_length, mod, key)

            xi_vec[:bit_length] = x_vec
            q_solt_att_idx = encoded_attr_to_idx[att_name]
            _add_scaled_kron_mod(tk_vec, main_basis[q_solt_att_idx], xi_vec, beta, mod)
            used_attr[q_solt_att_idx] = 1

    for i in range(len(encoded_attrs)):
        if used_attr[i]:
            continue
        alpha = random.randrange(1, mod)
        _add_scaled_kron_mod(tk_vec, main_basis[i], y_vec, alpha, mod)

    return tk_vec




def Search(
    c_a,
    c_b,
    token_a,
    token_b,
    mod: int,
) -> Tuple[List[Tuple[str, str]], int]:
    """
    始终让较小的一侧建桶。
    返回的 matches 仍保持 (pk_a, pk_b) 的顺序。
    """
    bucket_a: Dict[int, List[str]] = defaultdict(list)

    token_a = np.asarray(token_a, dtype=np.int64)
    token_b = np.asarray(token_b, dtype=np.int64)

    for row in c_a:
        score = decrypt(token_a, row["c"], mod)
        bucket_a[score].append(row["pk"])

    matches: List[Tuple[str, str]] = []
    for row in c_b:
        score = decrypt(token_b, row["c"], mod)
        for pk_a in bucket_a.get(score, []):
            matches.append((pk_a, row["pk"]))

    return matches, len(matches)



if __name__ == "__main__":

    p = 998244353
    # l = 5
    # O = orthogonal(l,p)
    # print(O)
    #
    # for idx, v in enumerate(O, start=1):
    #     print(f"o_{idx} =", v)
    # print("is orthonormal:", check_orthonormal(O))
    # o1 = np.asarray(O[2], dtype=object)
    # o2 = np.asarray(O[4], dtype=object)
    # print(int(np.dot(o1, o2) % p))
    bit_length = 20
    p = 3015000013

    table_a_file = "p.tbl"
    table_b_file = "ps.tbl"

    join_attrs_a = ["PARTKEY"]
    join_attrs_b = ["PARTKEY"]

    pk_attr_a = "PARTKEY"
    pk_attr_b = "PARTKEY"

    q_attr_a = "PARTKEY"
    q_attr_b = "PARTKEY"

    join_attrs_a_sorce = ["J-PARTKEY"]
    join_attrs_b_sorce = ["J-PARTKEY"]

    x_a = [["MFGR"], [["Manufacturer#3           "]]]
    # x_a = [["BRAND", "CONTAINER"], [["Brand#13"], ["SM CASE"]]]
    # 或
    # x_c = [
    #     ["BRAND", "CONTAINER", "SIZE"],
    #     [["Brand#13"], ["SM CASE", "SM BOX", "SM PACK", "SM PKG"], ["1", "2", "3", "4", "5"]]
    # ]
    x_b = None

    table_a = read_table(table_a_file, '|')
    table_b = read_table(table_b_file, '|')

    time1 = time.perf_counter()

    aux_basis_t = np.array([random.randrange(1,p) for _ in range(bit_length)], dtype=np.int64)
    aux_basis_d = np.array([random.randrange(1,p) for _ in range(AUX_DIM)], dtype=np.int64)
    key = secrets.token_hex(32)
    # key_t = secrets.token_hex(32)
    # key_d = secrets.token_hex(32)
    # aux_basis_t = derive_aux_basis(key_t, bit_length, p)
    # aux_basis_d = derive_aux_basis(key_d, AUX_DIM, p)
    encoded_attrs_a, main_basis_a = Setup(table_a, join_attrs_a, p)
    encoded_attrs_b, main_basis_b = Setup(table_b, join_attrs_b, p)
    time2 = time.perf_counter()





    print("setuptime:", str(time2 - time1))

    table_a_Y_att = join_attrs_a_sorce + encoded_attrs_a
    table_b_Y_att = join_attrs_b + encoded_attrs_b
    print("table_a_att", table_a_Y_att)
    print("table_b_att", table_b_Y_att)

    l_a = len(table_a)
    l_b = len(table_b)
    print("table_a_length", l_a)
    print("table_b_length", l_b)

    time1 = time.perf_counter()
    # aux_basis_t = derive_aux_basis(key_t, bit_length, p)
    # aux_basis_d = derive_aux_basis(key_d, AUX_DIM, p)
    C_a = Encrypt(table_a, encoded_attrs_a, join_attrs_a, pk_attr_a, main_basis_a, aux_basis_t, aux_basis_d, bit_length,
                  p, key)
    C_b = Encrypt(table_b, encoded_attrs_b, join_attrs_b, pk_attr_b, main_basis_b, aux_basis_t, aux_basis_d, bit_length,
                  p, key)
    time2 = time.perf_counter()
    print("enctime:", str(time2 - time1))

    time1 = time.perf_counter()
    shared_join_scalar = random.randrange(1, p)

    tk_a = TokenGen(join_attrs_a, encoded_attrs_a, x_a, main_basis_a, aux_basis_t, aux_basis_d, bit_length, p,
                    shared_join_scalar, q_attr_a, key)
    tk_b = TokenGen(join_attrs_b, encoded_attrs_b, x_b, main_basis_b, aux_basis_t, aux_basis_d, bit_length, p,
                    shared_join_scalar, q_attr_b, key)
    time2 = time.perf_counter()
    print("tokentime:", str(time2 - time1))

    time1 = time.perf_counter()
    matches, total = Search(C_a, C_b, tk_a, tk_b, p)
    time2 = time.perf_counter()
    print("search:", str(time2 - time1))
    print(matches)
    print(total)

