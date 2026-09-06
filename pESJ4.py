import pickle
import sys,os,time
import newtESJ as base
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

def storage_dtype_for_mod(p: int):
    if p <= np.iinfo(np.uint32).max:
        return np.uint32
    elif p <= np.iinfo(np.uint64).max:
        return np.uint64
    else:
        raise ValueError("p 太大，当前代码里没有合适的定长整数类型用于紧凑存储")

def read_table(path: str, delimiter: str = "|") -> List[Dict[str, str]]:
  """读取 .tbl，并去掉 DictReader 因行尾分隔符产生的 None 列。"""
  with open(path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter=delimiter, skipinitialspace=False)
    rows: List[Dict[str, str]] = []
    for row in reader:
      clean_row = {k: str(v) for k, v in row.items() if k is not None}
      rows.append(clean_row)
  return rows

def experiment_1(bit_length, iters):
  print("--------------------------- PRFlength="+str(bit_length))
  # table_a_file = "./table_a.in"
  # a="buyer_id"
  # pk_a="buyer_id"
  # x="phone_num"

  table_a_file = "data-part/1.tbl"
  join_attrs_a = ["PARTKEY"]
  pk_attr_a = "PARTKEY"

  join_attrs_a_sorce = ["J-PARTKEY"]
  q_attr_a = "PARTKEY"

  x_a = [["MFGR"],[["Manufacturer#3           "]]]

  table_a = read_table(table_a_file, '|')
  l_a=len(table_a)
  print(f"table_a_length: {l_a}")

  setup_time=[]
  enc_time=[]
  token_time=[]
  search_time=[]
  enc_timeper=[]
  search_timeper= []

  for i in range(iters):

    time1=time.perf_counter()
    aux_basis_t = np.array([random.randrange(1,p) for _ in range(bit_length)], dtype=np.int64)
    aux_basis_d = np.array([random.randrange(1,p) for _ in range(AUX_DIM)], dtype=np.int64)
    key = secrets.token_hex(32)

    encoded_attrs_a, main_basis_a = base.Setup(table_a, join_attrs_a, p)

    time2=time.perf_counter()
    setup_time.append(time2 - time1)

    time1 = time.perf_counter()
    C_a = base.Encrypt(table_a, encoded_attrs_a, join_attrs_a, pk_attr_a, main_basis_a, aux_basis_t, aux_basis_d, bit_length,
                  p, key)
    time2 = time.perf_counter()
    enc_time.append(time2-time1)
    enc_timeper.append((time2 - time1)/l_a)

    table_a_Y_att = join_attrs_a_sorce + encoded_attrs_a

    STORE_DTYPE = storage_dtype_for_mod(p)
    key_size = (
            len(pickle.dumps(table_a_Y_att, protocol=pickle.HIGHEST_PROTOCOL))
            + len(pickle.dumps(key, protocol=pickle.HIGHEST_PROTOCOL))
            + np.asarray(aux_basis_t, dtype=STORE_DTYPE).nbytes
            + np.asarray(aux_basis_d, dtype=STORE_DTYPE).nbytes
            + np.asarray(main_basis_a, dtype=STORE_DTYPE).nbytes
    )
    # key_size = (
    #         len(pickle.dumps(table_a_Y_att, protocol=pickle.HIGHEST_PROTOCOL))
    #         + len(pickle.dumps(key, protocol=pickle.HIGHEST_PROTOCOL))
    #         + len(pickle.dumps(key_t, protocol=pickle.HIGHEST_PROTOCOL))
    #         + len(pickle.dumps(key_d, protocol=pickle.HIGHEST_PROTOCOL))
    #         + np.asarray(main_basis_a, dtype=STORE_DTYPE).nbytes
    # )
    print("key_size =", key_size)

    edb_size = sum(np.asarray(row["c"], dtype=STORE_DTYPE).nbytes for row in C_a)
    edb_sizeper = np.asarray(C_a[0]["c"], dtype=STORE_DTYPE).nbytes
    print("edb_size =", edb_size)
    print("edb_sizeper =", edb_sizeper)


    #--------------------------------------------
    time1 = time.perf_counter()
    shared_join_scalar = random.randrange(1, p)
    tk_a = base.TokenGen(join_attrs_a, encoded_attrs_a, x_a, main_basis_a, aux_basis_t, aux_basis_d, bit_length, p,
                    shared_join_scalar, q_attr_a, key)
    time2=time.perf_counter()
    token_time.append(time2-time1)

    token_size = np.asarray(tk_a, dtype=STORE_DTYPE).nbytes
    print("token_size =", token_size)

    #--------------------------------------------
    time1 = time.perf_counter()
    tk_a = np.asarray(tk_a, dtype=np.int64)
    bucket_a: Dict[int, List[str]] = defaultdict(list)

    for row in C_a:
      score = base.decrypt(tk_a, row["c"], p)
      bucket_a[score].append(row["pk"])

    time2 = time.perf_counter()
    search_time.append(time2 - time1)
    search_timeper.append((time2 - time1) / l_a)

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

  avg_encrypt_time_per_row = sum(enc_timeper) / len(enc_timeper)
  avg_search_time_per_row = sum(search_timeper) / len(search_timeper)

  print(f"setup time:  {avg_setup_time:.6f} s on average")
  print(f"enc time:    {avg_encrypt_time:.6f} s on average")
  print(f"token time:  {avg_token_time:.6f} s on average")
  print(f"search time: {avg_search_time:.6f} s on average")
  print(f"key_size:         {key_size} B")
  print(f"edb_size:         {edb_size} B")
  print(f"edb_size_per_row: {edb_sizeper} B")
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

  return (
    avg_setup_time,
    avg_encrypt_time,
    avg_token_time,
    avg_search_time,
    key_size,
    edb_size,
    avg_encrypt_time_per_row,
    avg_search_time_per_row,
    edb_sizeper,
  )


if __name__ == "__main__":
  # p = 998244353
  # p = 2147483647
  # p = 2305843009
  # p = 3000000019
  p = 3015000013
  l = [20,40,80,160]

  avg_setup_times = []
  avg_encrypt_times = []
  avg_token_times = []
  avg_search_times = []

  key_sizes_bytes = []
  edb_sizes_bytes = []

  avg_encrypt_time_per_row = []
  avg_search_time_per_row = []
  edb_size_per_row_bytes = []

  for bit_length in l:
    (
      avg_setup_time,
      avg_encrypt_time,
      avg_token_time,
      avg_search_time,
      key_size_bytes,
      edb_size_bytes,
      encrypt_time_per_row,
      search_time_per_row,
      edb_size_per_row,
    ) = experiment_1(bit_length, 1)

    avg_setup_times.append(avg_setup_time)
    avg_encrypt_times.append(avg_encrypt_time)
    avg_token_times.append(avg_token_time)
    avg_search_times.append(avg_search_time)

    key_sizes_bytes.append(key_size_bytes)
    edb_sizes_bytes.append(edb_size_bytes)

    avg_encrypt_time_per_row.append(encrypt_time_per_row)
    avg_search_time_per_row.append(search_time_per_row)
    edb_size_per_row_bytes.append(edb_size_per_row)
    gc.collect()

  print("\n==================== Summary ====================")
  print("PRF_length           =", l)
  print("avg_setup_times (s)     =", avg_setup_times)
  print("avg_encrypt_times (s)   =", avg_encrypt_times)
  print("avg_token_times (s)     =", avg_token_times)
  print("avg_search_times (s)    =", avg_search_times)
  print("key_sizes (B)           =", key_sizes_bytes)
  print("edb_sizes (B)           =", edb_sizes_bytes)
  print("encrypt_time_per_row (s)=", avg_encrypt_time_per_row)
  print("search_time_per_row (s) =", avg_search_time_per_row)
  print("edb_size_per_row (B)    =", edb_size_per_row_bytes)
