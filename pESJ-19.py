import pickle
import sys,os,time
from pathlib import Path
import ESJ as base
from collections import defaultdict, Counter
import gc
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(1, os.path.abspath('..'))
#from pympler import asizeof
import numpy as np
import csv
import sys, os, math, random
from mpmath import mp
import mnum
import secrets
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
MAX_BASIS_TO_SUM = 32
AUX_DIM = 2
DOT_CHUNK = 1024

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
    参数固定为：
      QUANTITY1 = 1,  QUANTITY2 = 10, QUANTITY3 = 20
      BRAND1    = Brand#12
      BRAND2    = Brand#23
      BRAND3    = Brand#34
    返回值：
      real_matches: List[Tuple[str, str]]
      每个元素是 (part_row_partkey, lineitem_row_partkey)
      保留重复项，便于和密文 matches 做多重集比较。
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

def experiment_1(sf, iters):
  print("--------------------------- sf="+sf)
  BASE_DIR = Path(__file__).resolve().parent  # 当前 py 文件所在目录：Zp_ESJ
  q19_dir = BASE_DIR.parent / "mdata" / "Q19"  # 上一级 -> mdata -> Q19
  # table_a_file = "mdata/Q19/q_" + sf + "/part.tbl"

  table_a_file = q19_dir / ("q_" + sf) / "part.tbl"
  join_attrs_a = ["PARTKEY"]
  pk_attr_a = "PARTKEY"

  join_attrs_a_sorce = ["J-PARTKEY"]
  q_attr_a = "PARTKEY"
  x_c_1 = [["BRAND", "CONTAINER", "SIZE"],
           [["Brand#12"], ["SM CASE", "SM BOX", "SM PACK", "SM PKG"],
            ["1", "2", "3", "4", "5"]]]

  x_c_2 = [["BRAND", "CONTAINER", "SIZE"],
           [["Brand#23"], ["MED BAG", "MED BOX", "MED PACK", "MED PKG"],
            ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]]]

  x_c_3 = [["BRAND", "CONTAINER", "SIZE"],
           [["Brand#34"], ["LG CASE", "LG BOX", "LG PACK", "LG PKG"],
            ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
             "11", "12", "13", "14", "15"]]]
  x_a = []

  x_a.append(x_c_1)
  x_a.append(x_c_2)
  x_a.append(x_c_3)

  table_b_file = q19_dir / ("q_" + sf) / "lineitem.tbl"
  join_attrs_b = ["PARTKEY"]
  pk_attr_b = "PARTKEY"

  join_attrs_b_sorce = ["J-PARTKEY"]
  q_attr_b = "PARTKEY"
  y_c_1 = [["QUANTITY", "SHIPMODE", "SHIPINSTRUCT"],
           [[str(v) for v in range(1, 12)], ["AIR", "AIR REG"], ["DELIVER IN PERSON"]]]

  y_c_2 = [["QUANTITY", "SHIPMODE", "SHIPINSTRUCT"],
           [[str(v) for v in range(10, 21)], ["AIR", "AIR REG"], ["DELIVER IN PERSON"]]]

  y_c_3 = [["QUANTITY", "SHIPMODE", "SHIPINSTRUCT"],
           [[str(v) for v in range(20, 31)], ["AIR", "AIR REG"], ["DELIVER IN PERSON"]]]

  x_b = []
  x_b.append(y_c_1)
  x_b.append(y_c_2)
  x_b.append(y_c_3)

  table_a = read_table(table_a_file, '|')
  table_b = read_table(table_b_file, '|')
  l_a = len(table_a)
  l_b = len(table_b)
  print(f"table_a_length: {l_a}")
  print(f"table_b_length: {l_b}")

  setup_time = []
  enc_time = []
  token_time = []
  search_time = []

  for i in range(iters):

    time1 = time.perf_counter()

    aux_basis_t = np.array([random.randrange(1, p) for _ in range(bit_length)], dtype=np.int64)
    aux_basis_d = np.array([random.randrange(1, p) for _ in range(AUX_DIM)], dtype=np.int64)
    key = secrets.token_hex(32)

    encoded_attrs_a, main_basis_a = base.Setup(table_a, join_attrs_a, p)
    encoded_attrs_b, main_basis_b = base.Setup(table_b, join_attrs_b, p)
    time2 = time.perf_counter()
    setup_time.append(time2 - time1)

    time1 = time.perf_counter()
    C_a = base.Encrypt(table_a, encoded_attrs_a, join_attrs_a, pk_attr_a, main_basis_a, aux_basis_t, aux_basis_d, bit_length,
                  p, key)
    C_b = base.Encrypt(table_b, encoded_attrs_b, join_attrs_b, pk_attr_b, main_basis_b, aux_basis_t, aux_basis_d, bit_length,
                  p, key)
    time2 = time.perf_counter()
    enc_time.append((time2 - time1))



    timet1 = 0
    timet2 = 0
    shared_join_scalar = random.randrange(1, p)
    all_matches = []
    for i in range(len(x_a)):
      print("clause:",i+1)
      time1 = time.perf_counter()
      tk_a = base.TokenGen(join_attrs_a, encoded_attrs_a, x_a[i], main_basis_a, aux_basis_t, aux_basis_d, bit_length, p,
                           shared_join_scalar, q_attr_a, key)
      tk_b = base.TokenGen(join_attrs_b, encoded_attrs_b, x_b[i], main_basis_b, aux_basis_t, aux_basis_d, bit_length, p,
                           shared_join_scalar, q_attr_b, key)

      time2 = time.perf_counter()
      timet1 += (time2 - time1)

      #-----------------------------------------------------
      time1 = time.perf_counter()
      matches_i, total_i = base.Search(C_a, C_b, tk_a, tk_b, p)
      time2 = time.perf_counter()
      timet2 += (time2 - time1)

      print("total", total_i)
      all_matches.extend(matches_i)

    token_time.append(timet1)
    search_time.append(timet2)
    print(len(all_matches))
    real_matches = build_real_q19_matches(table_a, table_b)

    enc_counter = Counter(all_matches)
    real_counter = Counter(real_matches)

    extra = []
    for pair, cnt in enc_counter.items():
        diff = cnt - real_counter.get(pair, 0)
        if diff > 0:
            extra.extend([pair] * diff)

    print("extra count =", len(extra))
    print("extra sample =", extra[:20])

    del C_a
    del tk_a
    del encoded_attrs_a
    del main_basis_a
    del aux_basis_t
    del aux_basis_d
    gc.collect()

  avg_setup_time = sum(setup_time) / len(setup_time)
  avg_encrypt_time = sum(enc_time) / len(enc_time)
  avg_token_time = sum(token_time) / len(token_time)
  avg_search_time = sum(search_time) / len(search_time)


  print(f"setup time:  {avg_setup_time:.6f} s on average")
  print(f"enc time:    {avg_encrypt_time:.6f} s on average")
  print(f"token time:  {avg_token_time:.6f} s on average")
  print(f"search time: {avg_search_time:.6f} s on average")
  print()

  # -------------释放无用数据
  for var_name in [
    "table_a",
    "C_a",
    "tk_a",
    "encoded_attrs_a",
    "main_basis_a",
    "aux_basis_t",
    "aux_basis_d",
  ]:
    if var_name in locals():
      del locals()[var_name]

  gc.collect()

  return (avg_setup_time, avg_encrypt_time, avg_token_time, avg_search_time,)


if __name__ == "__main__":
    # p = 998244353
    # p = 2147483647
    # p = 2305843009
    # p = 3000000019
    p = 3015000013
    bit_length = 32

    scale_factors = ["20k", "40k", "60k", "80k", "100k"]

    avg_setup_times = []
    avg_encrypt_times = []
    avg_token_times = []
    avg_search_times = []


    for scale_factor in scale_factors:
        (
            avg_setup_time,
            avg_encrypt_time,
            avg_token_time,
            avg_search_time,
        ) = experiment_1(scale_factor, 1)
        avg_setup_times.append(avg_setup_time)
        avg_encrypt_times.append(avg_encrypt_time)
        avg_token_times.append(avg_token_time)
        avg_search_times.append(avg_search_time)

        gc.collect()

    print("\n==================== Summary ====================")
    print("scale_factors           =", scale_factors)
    print("avg_setup_times (s)     =", avg_setup_times)
    print("avg_encrypt_times (s)   =", avg_encrypt_times)
    print("avg_token_times (s)     =", avg_token_times)
    print("avg_search_times (s)    =", avg_search_times)

