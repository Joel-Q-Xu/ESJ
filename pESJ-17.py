import pickle
import sys,os,time
from pathlib import Path
import ESJ as base
from collections import defaultdict
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
  x_a = [["BRAND", "CONTAINER"],
           [["Brand#23"], ["MED BOX"]]]

  table_b_file = q19_dir / ("q_" + sf) / "lineitem.tbl"
  join_attrs_b = ["PARTKEY"]
  pk_attr_b = "PARTKEY"

  join_attrs_b_sorce = ["J-PARTKEY"]
  q_attr_b = "PARTKEY"
  x_b = None

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
  # setup_time_b = []
  # enc_time_b = []
  # token_time_b = []
  # search_time_b = []

  for i in range(iters):

    time1 = time.perf_counter()

    # aux_basis_t = np.array([random.randrange(1, p) for _ in range(bit_length)], dtype=np.int64)
    # aux_basis_d = np.array([random.randrange(1, p) for _ in range(AUX_DIM)], dtype=np.int64)
    key = secrets.token_hex(32)
    key_t = secrets.token_hex(32)
    key_d = secrets.token_hex(32)

    encoded_attrs_a, main_basis_a = base.Setup(table_a, join_attrs_a, p)
    encoded_attrs_b, main_basis_b = base.Setup(table_b, join_attrs_b, p)
    time2 = time.perf_counter()
    setup_time.append(time2 - time1)

    time1 = time.perf_counter()
    aux_basis_t = base.derive_aux_basis(key_t, bit_length, p)
    aux_basis_d = base.derive_aux_basis(key_d, AUX_DIM, p)
    C_a = base.Encrypt(table_a, encoded_attrs_a, join_attrs_a, pk_attr_a, main_basis_a, aux_basis_t, aux_basis_d, bit_length,
                  p, key)
    C_b = base.Encrypt(table_b, encoded_attrs_b, join_attrs_b, pk_attr_b, main_basis_b, aux_basis_t, aux_basis_d, bit_length,
                  p, key)
    time2 = time.perf_counter()
    enc_time.append((time2 - time1))


    time1 = time.perf_counter()
    shared_join_scalar = random.randrange(1, p)
    aux_basis_t = base.derive_aux_basis(key_t, bit_length, p)
    aux_basis_d = base.derive_aux_basis(key_d, AUX_DIM, p)
    tk_a = base.TokenGen(join_attrs_a, encoded_attrs_a, x_a, main_basis_a, aux_basis_t, aux_basis_d, bit_length, p,
                    shared_join_scalar, q_attr_a, key)
    tk_b = base.TokenGen(join_attrs_b, encoded_attrs_b, x_b, main_basis_b, aux_basis_t, aux_basis_d, bit_length, p,
                    shared_join_scalar, q_attr_b, key)
    time2 = time.perf_counter()

    token_time.append((time2 - time1))

    time1 = time.perf_counter()
    matches, total = base.Search(C_a, C_b, tk_a, tk_b, p)
    time2 = time.perf_counter()
    search_time.append((time2 - time1))
    print(total)
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
    # scale_factors = ["100k"]

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

