import sys, os, time, csv, random, secrets, gc
from pathlib import Path
from collections import Counter
from typing import Dict, List

import numpy as np
import newtESJ as base

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(1, os.path.abspath('..'))

MAX_BASIS_TO_SUM = 32
AUX_DIM = 2


def read_table(path: str, delimiter: str = "|") -> List[Dict[str, str]]:
    """读取 .tbl，并去掉 DictReader 因行尾分隔符产生的 None 列。"""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter, skipinitialspace=False)
        rows: List[Dict[str, str]] = []
        for row in reader:
            clean_row = {k: str(v) for k, v in row.items() if k is not None}
            rows.append(clean_row)
    return rows


def build_real_q19_matches(table_a, table_b):
    """
    按当前这组 Q19 条件，在明文上计算真实 join 结果。
    返回值保留重复项，便于和密文 matches 做多重集比较。
    """
    part_by_pk = {row["PARTKEY"]: row for row in table_a}
    real_matches = []

    for row_b in table_b:
        pk = row_b["PARTKEY"]
        row_a = part_by_pk.get(pk)
        if row_a is None:
            continue

        brand = row_a["BRAND"]
        container = row_a["CONTAINER"]
        size = int(row_a["SIZE"])

        quantity = int(row_b["QUANTITY"])
        shipmode = row_b["SHIPMODE"]
        shipinstruct = row_b["SHIPINSTRUCT"]

        common_ok = (
            shipmode in ("AIR", "AIR REG")
            and shipinstruct == "DELIVER IN PERSON"
        )

        cond1 = (
            brand == "Brand#12"
            and container in ("SM CASE", "SM BOX", "SM PACK", "SM PKG")
            and 1 <= quantity <= 11
            and 1 <= size <= 5
        )

        cond2 = (
            brand == "Brand#23"
            and container in ("MED BAG", "MED BOX", "MED PACK", "MED PKG")
            and 10 <= quantity <= 20
            and 1 <= size <= 10
        )

        cond3 = (
            brand == "Brand#34"
            and container in ("LG CASE", "LG BOX", "LG PACK", "LG PKG")
            and 20 <= quantity <= 30
            and 1 <= size <= 15
        )

        if common_ok and (cond1 or cond2 or cond3):
            real_matches.append((row_a["PARTKEY"], row_b["PARTKEY"]))

    return real_matches


def query_q17(
    C_a, C_b,
    encoded_attrs_a, encoded_attrs_b,
    main_basis_a, main_basis_b,
    aux_basis_t, aux_basis_d,
    join_attrs_a, join_attrs_b,
    q_attr_a, q_attr_b,
    bit_length, p, key,
    table_a
):
    """
    第一种查询，输出名为 Q17
    """
    x_a = [
        ["BRAND", "CONTAINER"],
        [["Brand#23"], ["MED BOX"]]
    ]
    x_b = None

    shared_join_scalar = random.randrange(1, p)

    t1 = time.perf_counter()
    tk_a = base.TokenGen(
        join_attrs_a, encoded_attrs_a, x_a, main_basis_a,
        aux_basis_t, aux_basis_d, bit_length, p,
        shared_join_scalar, q_attr_a, key
    )
    tk_b = base.TokenGen(
        join_attrs_b, encoded_attrs_b, x_b, main_basis_b,
        aux_basis_t, aux_basis_d, bit_length, p,
        shared_join_scalar, q_attr_b, key
    )
    t2 = time.perf_counter()
    token_time = t2 - t1

    t1 = time.perf_counter()
    matches, total = base.Search(C_a, C_b, tk_a, tk_b, p)
    t2 = time.perf_counter()
    search_time = t2 - t1

    qualifying_pk = {
        row["PARTKEY"]
        for row in table_a
        if row["BRAND"] == "Brand#23" and row["CONTAINER"] == "MED BOX"
    }

    extra = [
        (pk_a, pk_b)
        for pk_a, pk_b in matches
        if not (pk_a == pk_b and pk_a in qualifying_pk)
    ]

    return {
        "token_time": token_time,
        "search_time": search_time,
        "total": total,
        "match_count": len(matches),
        "extra_count": len(extra),
        "extra_sample": extra[:20],
    }


def query_q19(
    C_a, C_b,
    encoded_attrs_a, encoded_attrs_b,
    main_basis_a, main_basis_b,
    aux_basis_t, aux_basis_d,
    join_attrs_a, join_attrs_b,
    q_attr_a, q_attr_b,
    bit_length, p, key,
    table_a, table_b
):
    """
    第二种查询，输出名为 Q19
    """
    q19_a_clauses = [
        [["BRAND", "CONTAINER", "SIZE"],
         [["Brand#12"], ["SM CASE", "SM BOX", "SM PACK", "SM PKG"],
          ["1", "2", "3", "4", "5"]]],
        [["BRAND", "CONTAINER", "SIZE"],
         [["Brand#23"], ["MED BAG", "MED BOX", "MED PACK", "MED PKG"],
          ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]]],
        [["BRAND", "CONTAINER", "SIZE"],
         [["Brand#34"], ["LG CASE", "LG BOX", "LG PACK", "LG PKG"],
          ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
           "11", "12", "13", "14", "15"]]],
    ]

    q19_b_clauses = [
        [["QUANTITY", "SHIPMODE", "SHIPINSTRUCT"],
         [[str(v) for v in range(1, 12)], ["AIR", "AIR REG"], ["DELIVER IN PERSON"]]],
        [["QUANTITY", "SHIPMODE", "SHIPINSTRUCT"],
         [[str(v) for v in range(10, 21)], ["AIR", "AIR REG"], ["DELIVER IN PERSON"]]],
        [["QUANTITY", "SHIPMODE", "SHIPINSTRUCT"],
         [[str(v) for v in range(20, 31)], ["AIR", "AIR REG"], ["DELIVER IN PERSON"]]],
    ]

    shared_join_scalar = random.randrange(1, p)

    token_time = 0.0
    search_time = 0.0
    all_matches = []
    clause_totals = []

    for idx in range(len(q19_a_clauses)):
        t1 = time.perf_counter()
        tk_a = base.TokenGen(
            join_attrs_a, encoded_attrs_a, q19_a_clauses[idx], main_basis_a,
            aux_basis_t, aux_basis_d, bit_length, p,
            shared_join_scalar, q_attr_a, key
        )
        tk_b = base.TokenGen(
            join_attrs_b, encoded_attrs_b, q19_b_clauses[idx], main_basis_b,
            aux_basis_t, aux_basis_d, bit_length, p,
            shared_join_scalar, q_attr_b, key
        )
        t2 = time.perf_counter()
        token_time += (t2 - t1)

        t1 = time.perf_counter()
        matches_i, total_i = base.Search(C_a, C_b, tk_a, tk_b, p)
        t2 = time.perf_counter()
        search_time += (t2 - t1)

        clause_totals.append(total_i)
        all_matches.extend(matches_i)

    real_matches = build_real_q19_matches(table_a, table_b)

    enc_counter = Counter(all_matches)
    real_counter = Counter(real_matches)

    extra = []
    for pair, cnt in enc_counter.items():
        diff = cnt - real_counter.get(pair, 0)
        if diff > 0:
            extra.extend([pair] * diff)

    return {
        "token_time": token_time,
        "search_time": search_time,
        "total": len(all_matches),
        "match_count": len(all_matches),
        "clause_totals": clause_totals,
        "extra_count": len(extra),
        "extra_sample": extra[:20],
    }


def run_both_queries_once_shared_cipher(sf, iters):
    print("--------------------------- sf=" + sf)

    BASE_DIR = Path(__file__).resolve().parent

    # 同一份数据源：Q17 / Q19 都复用这一份表
    shared_data_dir = BASE_DIR.parent / "mdata" / "Q19"
    table_a_file = shared_data_dir / ("q_" + sf) / "part.tbl"
    table_b_file = shared_data_dir / ("q_" + sf) / "lineitem.tbl"

    join_attrs_a = ["PARTKEY"]
    pk_attr_a = "PARTKEY"
    q_attr_a = "PARTKEY"

    join_attrs_b = ["PARTKEY"]
    pk_attr_b = "PARTKEY"
    q_attr_b = "PARTKEY"

    # 数据只读一次
    table_a = read_table(table_a_file, '|')
    table_b = read_table(table_b_file, '|')

    print(f"table_a_length: {len(table_a)}")
    print(f"table_b_length: {len(table_b)}")

    setup_times = []
    encrypt_times = []

    q17_token_times = []
    q17_search_times = []

    q19_token_times = []
    q19_search_times = []

    for _ in range(iters):
        # ==================== shared setup ====================
        t1 = time.perf_counter()
        encoded_attrs_a, main_basis_a = base.Setup(table_a, join_attrs_a, p)
        encoded_attrs_b, main_basis_b = base.Setup(table_b, join_attrs_b, p)
        t2 = time.perf_counter()
        setup_times.append(t2 - t1)

        # ==================== shared encrypt ====================
        aux_basis_t = np.array([random.randrange(1, p) for _ in range(t)], dtype=np.int64)
        aux_basis_d = np.array([random.randrange(1, p) for _ in range(AUX_DIM)], dtype=np.int64)
        key = secrets.token_hex(32)

        t1 = time.perf_counter()
        C_a = base.Encrypt(
            table_a, encoded_attrs_a, join_attrs_a, pk_attr_a,
            main_basis_a, aux_basis_t, aux_basis_d, t, p, key
        )
        C_b = base.Encrypt(
            table_b, encoded_attrs_b, join_attrs_b, pk_attr_b,
            main_basis_b, aux_basis_t, aux_basis_d, t, p, key
        )
        t2 = time.perf_counter()
        encrypt_times.append(t2 - t1)

        # ==================== Q17 ====================
        q17 = query_q17(
            C_a, C_b,
            encoded_attrs_a, encoded_attrs_b,
            main_basis_a, main_basis_b,
            aux_basis_t, aux_basis_d,
            join_attrs_a, join_attrs_b,
            q_attr_a, q_attr_b,
            t, p, key,
            table_a
        )
        q17_token_times.append(q17["token_time"])
        q17_search_times.append(q17["search_time"])

        print("[Q17]")
        print("total =", q17["total"])
        print("extra count =", q17["extra_count"])
        print("extra sample =", q17["extra_sample"])

        # ==================== Q19 ====================
        q19 = query_q19(
            C_a, C_b,
            encoded_attrs_a, encoded_attrs_b,
            main_basis_a, main_basis_b,
            aux_basis_t, aux_basis_d,
            join_attrs_a, join_attrs_b,
            q_attr_a, q_attr_b,
            t, p, key,
            table_a, table_b
        )
        q19_token_times.append(q19["token_time"])
        q19_search_times.append(q19["search_time"])

        print("[Q19]")
        print("clause totals =", q19["clause_totals"])
        print("total =", q19["total"])
        print("extra count =", q19["extra_count"])
        print("extra sample =", q19["extra_sample"])

        # ==================== release ====================
        del C_a, C_b
        del encoded_attrs_a, encoded_attrs_b
        del main_basis_a, main_basis_b
        del aux_basis_t, aux_basis_d
        gc.collect()

    avg_setup_time = sum(setup_times) / len(setup_times)
    avg_encrypt_time = sum(encrypt_times) / len(encrypt_times)

    avg_q17_token_time = sum(q17_token_times) / len(q17_token_times)
    avg_q17_search_time = sum(q17_search_times) / len(q17_search_times)

    avg_q19_token_time = sum(q19_token_times) / len(q19_token_times)
    avg_q19_search_time = sum(q19_search_times) / len(q19_search_times)

    print()
    print("setup time(shared):  {:.6f} s on average".format(avg_setup_time))
    print("enc time(shared):    {:.6f} s on average".format(avg_encrypt_time))
    print("Q17 token time:      {:.6f} s on average".format(avg_q17_token_time))
    print("Q17 search time:     {:.6f} s on average".format(avg_q17_search_time))
    print("Q19 token time:      {:.6f} s on average".format(avg_q19_token_time))
    print("Q19 search time:     {:.6f} s on average".format(avg_q19_search_time))
    print()

    return (
        avg_setup_time,
        avg_encrypt_time,
        avg_q17_token_time,
        avg_q17_search_time,
        avg_q19_token_time,
        avg_q19_search_time,
    )


if __name__ == "__main__":
    p = 3015000013
    t=20

    scale_factors = ["20k", "40k", "60k", "80k", "100k"]
    # scale_factors = ["100k"]

    avg_setup_times = []
    avg_encrypt_times = []
    avg_q17_token_times = []
    avg_q17_search_times = []
    avg_q19_token_times = []
    avg_q19_search_times = []

    for scale_factor in scale_factors:
        (
            avg_setup_time,
            avg_encrypt_time,
            avg_q17_token_time,
            avg_q17_search_time,
            avg_q19_token_time,
            avg_q19_search_time,
        ) = run_both_queries_once_shared_cipher(scale_factor, 1)

        avg_setup_times.append(avg_setup_time)
        avg_encrypt_times.append(avg_encrypt_time)
        avg_q17_token_times.append(avg_q17_token_time)
        avg_q17_search_times.append(avg_q17_search_time)
        avg_q19_token_times.append(avg_q19_token_time)
        avg_q19_search_times.append(avg_q19_search_time)

        gc.collect()

    print("\n==================== Summary ====================")
    print("scale_factors             =", scale_factors)
    print("avg_setup_times (shared)  =", avg_setup_times)
    print("avg_encrypt_times (shared)=", avg_encrypt_times)
    print("avg_q17_token_times (s)   =", avg_q17_token_times)
    print("avg_q17_search_times (s)  =", avg_q17_search_times)
    print("avg_q19_token_times (s)   =", avg_q19_token_times)
    print("avg_q19_search_times (s)  =", avg_q19_search_times)
