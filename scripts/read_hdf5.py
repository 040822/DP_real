#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用法：
  python dump_h5.py                    # 默认读取 ./two_panda_ee.h5，输出到 data/
  python dump_h5.py --input ./two_panda_ee.h5 --out data --mode summary
  python dump_h5.py --mode full        # 尝试导出尽量多的数据（大数组会抽样）

输出：
  data/two_panda_ee_summary.json
  data/two_panda_ee_tree.txt
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Union

import h5py
import numpy as np

# ---- 配置：控制“full”模式下的数组阈值，超过就抽样 ----
MAX_ELEMENTS_FULL = 200_000   # 超过则抽样
SAMPLE_FIRST_AXIS = 5         # 抽样时，沿第0维取前N个

def np_to_native(x: np.ndarray) -> Union[list, int, float, str, None]:
    """把 numpy 数据转成 JSON 可序列化的 Python 原生类型。
    大数组会在 full 模式下抽样；summary 模式下不会调用到这个函数。
    """
    if x.ndim == 0:
        return x.item()
    # 太大就抽样
    if x.size > MAX_ELEMENTS_FULL and x.ndim >= 1:
        take = min(SAMPLE_FIRST_AXIS, x.shape[0])
        return {
            "__sample__": True,
            "sample_shape": (take,) + x.shape[1:],
            "orig_shape": x.shape,
            "data": x[:take].tolist()
        }
    return x.tolist()

def attrs_to_native(attrs: h5py.AttributeManager) -> Dict[str, Any]:
    out = {}
    for k in attrs.keys():
        v = attrs[k]
        if isinstance(v, np.ndarray):
            if v.ndim == 0:
                out[k] = v.item()
            else:
                # 属性一般很小，直接转
                out[k] = v.tolist()
        elif isinstance(v, (bytes, bytearray)):
            out[k] = v.decode("utf-8", errors="ignore")
        else:
            out[k] = v
    return out

def build_summary_node(name: str, obj: Union[h5py.Group, h5py.Dataset], mode: str) -> Dict[str, Any]:
    """递归构建可 JSON 化的描述"""
    if isinstance(obj, h5py.Dataset):
        node = {
            "__type": "dataset",
            "name": name.split("/")[-1],
            "path": name,
            "shape": tuple(obj.shape),
            "dtype": str(obj.dtype),
            "attrs": attrs_to_native(obj.attrs)
        }
        if mode == "full":
            try:
                data = obj[()]  # 读取全部
                if isinstance(data, np.ndarray):
                    node["data"] = np_to_native(data)
                else:
                    node["data"] = data
            except Exception as e:
                node["data_error"] = f"{type(e).__name__}: {e}"
        return node
    else:
        # Group
        node = {
            "__type": "group",
            "name": name.split("/")[-1] if name else "/",
            "path": name if name else "/",
            "attrs": attrs_to_native(obj.attrs),
            "children": {}
        }
        for key in obj.keys():
            child = obj[key]
            child_name = f"{name}/{key}" if name else f"/{key}"
            node["children"][key] = build_summary_node(child_name, child, mode)
        return node

def build_tree_text(name: str, obj: Union[h5py.Group, h5py.Dataset], prefix: str = "") -> str:
    """生成类似目录树的文本视图"""
    lines = []
    if isinstance(obj, h5py.Dataset):
        lines.append(f"{prefix}- {name.split('/')[-1]}  [dataset]  shape={obj.shape}, dtype={obj.dtype}")
    else:
        header = "/" if name in ("", "/") else name.split("/")[-1]
        lines.append(f"{prefix}+ {header} [group]")
        keys = list(obj.keys())
        for i, k in enumerate(keys):
            child = obj[k]
            new_prefix = prefix + ("  " if i == len(keys) - 1 else "  ")
            lines.append(build_tree_text(f"{name}/{k}" if name else f"/{k}", child, prefix + "  "))
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default="/root/autodl-tmp/basket_pick_up/episode_0.hdf5", help="H5 文件路径")
    parser.add_argument("--out", "-o", default="data", help="输出目录")
    parser.add_argument("--mode", choices=["summary", "full"], default="summary",
                        help="summary: 只导出结构与元信息；full: 尝试导出尽量多的数据（大数组抽样）")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = in_path.stem
    json_path = out_dir / f"{base}_summary.json"  # 名字沿用 summary；full 也用这个名字
    tree_path = out_dir / f"{base}_tree.txt"

    if not in_path.exists():
        raise FileNotFoundError(f"H5 文件不存在: {in_path}")

    with h5py.File(in_path, "r") as f:
        # JSON（summary / full）
        root_name = ""  # 根组用空字符串表示
        root_node = build_summary_node(root_name, f, args.mode)
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(root_node, jf, ensure_ascii=False, indent=2)

        # 文本树
        tree_text = build_tree_text("", f)
        with open(tree_path, "w", encoding="utf-8") as tf:
            tf.write(tree_text + "\n")

    print(f"已生成：{json_path}")
    print(f"已生成：{tree_path}")

if __name__ == "__main__":
    main()
