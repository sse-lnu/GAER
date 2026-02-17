from __future__ import annotations

import warnings

warnings.filterwarnings(
    "ignore",
    message=r"KMeans is known to have a memory leak on Windows with MKL.*",
)

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from eval.clusterer import ClusterAndEval
from models.node2vec_model import Node2VecModel
from preprocessing.gaer_data import HeterogeneousData
from preprocessing.negar_data import NEGARData
from train.gae_training import train as train_gae

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


DATASET_FILES = {
    "AS4": "archstudio.csv",
    "Bash": "bash.csv",
    "Chrom": "chromium.csv",
    "Hadoop": "hadoop.csv",
    "HDF": "hdf.csv",
    "HDC": "hdc.csv",
    "OODT": "oodt.csv",
    "Jabref": "jabref.csv",
    "TeamMates": "teammates.csv",
    "Libxml": "libxml.csv",
}

DEP_FILES = {
    "AS4": "archstudio_deps.csv",
    "Bash": "bash_deps.csv",
    "Chrom": "chromium_deps.csv",
    "Hadoop": "hadoop_deps.csv",
    "HDF": "hdf_deps.csv",
    "HDC": "hdc_deps.csv",
    "OODT": "oodt_deps.csv",
    "Jabref": "jabref_deps.csv",
    "TeamMates": "teammates_deps.csv",
    "Libxml": "libxml_deps.csv",
}

def _load_tables(data_dir: Path, names: List[str]) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    datasets: Dict[str, pd.DataFrame] = {}
    deps: Dict[str, pd.DataFrame] = {}

    for name in names:
        node_path = data_dir / DATASET_FILES[name]
        dep_path = data_dir / DEP_FILES[name]
        datasets[name] = pd.read_csv(node_path)
        deps[name] = pd.read_csv(dep_path)

    return datasets, deps


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_gaer_one(
    name: str,
    df: pd.DataFrame,
    df_dep: pd.DataFrame,
    ce: ClusterAndEval,
    encoder: str,
    epochs: int,
    hidden: int,
    dropout: float,
    lr: float,
    do_eval: bool,
    user_k: int | None,
    save_labels: bool,
) -> Tuple[pd.DataFrame, Dict]:
    t0 = time.perf_counter()
    data = HeterogeneousData(df, df_dep)
    t_build = time.perf_counter()

    Z, logs = train_gae(
        data,
        epochs=epochs,
        hidden=hidden,
        dropout=dropout,
        lr=lr,
        encoder=encoder,
    )
    t_train = time.perf_counter()

    df_row, out = ce.run(data=data, Z=Z, do_eval=do_eval, user_k=user_k)
    t_cluster = time.perf_counter()

    df_row.insert(0, "Dataset", name)
    df_row["Encoder"] = encoder
    df_row["GAE_loss"] = logs.get("loss")
    df_row["build_data_time(sec)"] = round(t_build - t0, 4)
    df_row["Train_time(sec)"] = round(t_train - t_build, 4)
    df_row["Cluster_time(sec)"] = round(t_cluster - t_train, 4)
    df_row["Total_time(sec)"] = round(t_cluster - t0, 4)

    if not save_labels:
        out.pop("labels", None)

    return df_row, out


def run_negar_one(
    name: str,
    df: pd.DataFrame,
    df_dep: pd.DataFrame,
    ce: ClusterAndEval,
    n2v: Node2VecModel,
    do_eval: bool,
    user_k: int | None,
    save_labels: bool,
) -> Tuple[pd.DataFrame, Dict]:
    t0 = time.perf_counter()

    if hasattr(NEGARData, "from_tables") and callable(getattr(NEGARData, "from_tables")):
        data = NEGARData.from_tables(df, df_dep, use_majority_vote=True)
    else:
        data = NEGARData(df, df_dep, use_majority_vote=True)

    t_build = time.perf_counter()

    Z = n2v.fit_transform(data)
    t_embed = time.perf_counter()

    df_row, out = ce.run(data=data, Z=Z, do_eval=do_eval, user_k=user_k)
    t_cluster = time.perf_counter()

    df_row.insert(0, "Dataset", name)
    df_row["Node2Vec_dim"] = n2v.dimensions
    df_row["build_data_time(sec)"] = round(t_build - t0, 4)
    df_row["Train_time(sec)"] = round(t_embed - t_build, 4)
    df_row["Cluster_time(sec)"] = round(t_cluster - t_embed, 4)
    df_row["Total_time(sec)"] = round(t_cluster - t0, 4)

    if not save_labels:
        out.pop("labels", None)

    return df_row, out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--pipeline", type=str, choices=["gaer", "negar", "both"], default="both")
    p.add_argument("--datasets", nargs="*", default=["all"])
    p.add_argument("--out_dir", type=str, default="results")
    p.add_argument("--save_labels", action="store_true")
    p.add_argument("--no_eval", action="store_true")
    p.add_argument("--k_min", type=int, default=10)
    p.add_argument("--k_max", type=int, default=30)
    p.add_argument("--sample_size", type=int, default=1000)
    p.add_argument("--user_k", type=int, default=None)

    # GAER defaults
    p.add_argument("--encoder", type=str, choices=["gat", "gcn"], default="gat")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=1e-4)

    # NEGAR defaults
    p.add_argument("--n2v_dim", type=int, default=128)
    p.add_argument("--n2v_walk_length", type=int, default=30)
    p.add_argument("--n2v_num_walks", type=int, default=200)
    p.add_argument("--n2v_window", type=int, default=15)
    p.add_argument("--n2v_epochs", type=int, default=5)
    p.add_argument("--n2v_p", type=float, default=1.0)
    p.add_argument("--n2v_q", type=float, default=1.0)
    p.add_argument("--n2v_negative", type=int, default=5)
    p.add_argument("--n2v_workers", type=int, default=4)

    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_names = list(DATASET_FILES.keys())
    names = all_names if (len(args.datasets) == 1 and args.datasets[0].lower() == "all") else args.datasets
    for n in names:
        if n not in DATASET_FILES:
            raise ValueError(f"Unknown dataset '{n}'. Known: {all_names}")

    datasets, deps = _load_tables(data_dir, names)

    ce = ClusterAndEval(
        k_range=range(int(args.k_min), int(args.k_max) + 1),
        sample_size=int(args.sample_size),
    )

    do_eval = not bool(args.no_eval)
    cpu_count = os.cpu_count() or 1
    safe_cap = max(1, min(4, cpu_count - 1))
    requested = int(args.n2v_workers)
    if requested <= 0:
        n2v_workers = safe_cap
    else:
        n2v_workers = max(1, min(requested, safe_cap))

    rows = []
    labels_dump = {}

    if args.pipeline in {"gaer", "both"}:
        iterator = tqdm(names, desc="GAER", unit="dataset") if tqdm else names
        for name in iterator:
            if not tqdm:
                print(f"[GAER] {name} ...")

            df_row, out = run_gaer_one(
                name=name,
                df=datasets[name],
                df_dep=deps[name],
                ce=ce,
                encoder=args.encoder,
                epochs=args.epochs,
                hidden=args.hidden,
                dropout=args.dropout,
                lr=args.lr,
                do_eval=do_eval,
                user_k=args.user_k,
                save_labels=args.save_labels,
            )
            rows.append(df_row)
            labels_dump[f"GAER::{name}"] = out

    if args.pipeline in {"negar", "both"}:
        iterator = tqdm(names, desc="NEGAR", unit="dataset") if tqdm else names
        for name in iterator:
            if not tqdm:
                print(f"[NEGAR] {name} ...")

            # FIX: name exists here, so compute size here
            n_points = int(datasets[name].shape[0])
            is_big = n_points > 5000

            # FIX: build Node2Vec per dataset (big ones get cheaper params)
            n2v = Node2VecModel(
                dimensions=int(args.n2v_dim),
                walk_length=int(args.n2v_walk_length),
                num_walks=100 if is_big else int(args.n2v_num_walks),
                window=int(args.n2v_window),
                epochs=1 if is_big else int(args.n2v_epochs),
                p=float(args.n2v_p),
                q=float(args.n2v_q),
                workers=n2v_workers,
                negative=1 if is_big else int(args.n2v_negative),
            )

            df_row, out = run_negar_one(
                name=name,
                df=datasets[name],
                df_dep=deps[name],
                ce=ce,
                n2v=n2v,
                do_eval=do_eval,
                user_k=args.user_k,
                save_labels=args.save_labels,
            )
            rows.append(df_row)
            labels_dump[f"NEGAR::{name}"] = out

    results = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    tag = _now_tag()
    out_csv = out_dir / f"results_{args.pipeline}_{tag}.csv"
    results.to_csv(out_csv, index=False)

    if args.save_labels:
        out_json = out_dir / f"labels_{args.pipeline}_{tag}.json"
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(labels_dump, f, indent=2)

    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
