from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import os
import re
import numpy as np
import pandas as pd
import torch

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, normalize
from torch_geometric.data import HeteroData

from preprocessing.w2v_embeddings import W2VEmbeddingGenerator


def infer_language_from_files(files: pd.Series) -> str:
    exts = (
        files.astype(str)
        .str.lower()
        .str.extract(r"(\.[a-z0-9]+)$", expand=False)
        .dropna()
    )
    if exts.empty:
        return "java"
    ext = exts.value_counts().idxmax()
    mapping = {
        ".java": "java",
        ".py": "python",
        ".cs": "csharp",
        ".c": "c",
        ".h": "c",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".cxx": "cpp",
        ".hpp": "cpp",
        ".hh": "cpp",
        ".js": "javascript",
        ".ts": "typescript",
        ".kt": "kotlin",
        ".go": "go",
        ".rb": "ruby",
        ".php": "php",
        ".swift": "swift",
    }
    return mapping.get(ext, "java")


from collections import Counter
import numpy as np
import pandas as pd


class HeterogeneousData(HeteroData):
    def __init__(self, df: pd.DataFrame, df_dep: pd.DataFrame, w2v_params: Optional[dict] = None, max_df: float = 0.9):
        super().__init__()
        self.df = df.copy()
        self.df_dep = df_dep.copy()
        self.w2v_params = w2v_params or dict(vector_size=100, window=5, min_count=5, sg=1, epochs=10, max_vocab_size=2000)
        self.max_df = float(max_df)

        self.label_encoder = None
        self.num_classes = 0
        self.relations = []

        self._clean_tables()
        self._create_node_features()
        self._create_edges()

    def _clean_tables(self) -> None:
        df = self.df.copy()
        dep = self.df_dep.copy()

        df["File"] = df["File"].astype(str).str.replace("\\", "/", regex=False)
        df = df[df["Module"] != "unmapped"]
        df["Module"] = (df["Module"].astype(str).str.strip().str.lower().replace({"nan": None, "none": None, "": None}))
    
        if "Entity" not in df.columns:
            df["Entity"] = df["File"].astype(str).str.replace("/", ".", regex=False)

        if "Code" not in df.columns:
            if "Member_Name" in df.columns:
                code_map = (
                    df.groupby("File")["Member_Name"]
                    .apply(lambda s: " ".join(sorted(set(str(x) for x in s if pd.notna(x) and str(x).strip()))))
                    .to_dict()
                )
                df["Code"] = df["File"].map(code_map).fillna("")
            else:
                df["Code"] = ""
        else:
            df["Code"] = df["Code"].fillna("").astype(str)

        nodes = (
            df.groupby("File", as_index=False)
              .agg(
                  Code=("Code", "first"),
                  Entity=("Entity", "first"),
                  Module_List=("Module", lambda s: sorted(set([x for x in s if x is not None]))),
              )
        )

        empty = nodes["Module_List"].apply(len) == 0
        if empty.any():
            nodes.loc[empty, "Module_List"] = [["__none__"]] * int(empty.sum())

        nodes["Duplicated"] = nodes["Module_List"].apply(len).gt(1)
        presence = Counter(m for mods in nodes["Module_List"] for m in mods)

        def pick_min_presence(mods):
            best = mods[-1]
            best_c = presence.get(best, 10**12)
            best_i = len(mods) - 1
            for i, m in enumerate(mods):
                c = presence.get(m, 10**12)
                if c < best_c or (c == best_c and i > best_i):
                    best, best_c, best_i = m, c, i
            return best

        nodes["Module"] = nodes["Module_List"].apply(pick_min_presence)

        nodes = nodes.reset_index(drop=True)
        nodes["File_ID"] = np.arange(len(nodes), dtype=int)
        self.df = nodes

        dep["Source_File"] = dep["Source_File"].astype(str).str.replace("\\", "/", regex=False)
        dep["Target_File"] = dep["Target_File"].astype(str).str.replace("\\", "/", regex=False)
        dep["Dependency_Count"] = pd.to_numeric(dep["Dependency_Count"], errors="coerce").fillna(1.0)

        valid_files = set(self.df["File"].astype(str))
        dep = dep[dep["Source_File"].isin(valid_files) & dep["Target_File"].isin(valid_files)]
        dep = dep[~dep["Dependency_Type"].str.contains("possible", case=False, na=False)]
        dep = dep.groupby(["Source_File", "Target_File", "Dependency_Type"], as_index=False)["Dependency_Count"].sum()

        idx_map = dict(zip(self.df["File"].astype(str), self.df["File_ID"].astype(int)))
        dep["Source_ID"] = dep["Source_File"].map(idx_map)
        dep["Target_ID"] = dep["Target_File"].map(idx_map)
        dep = dep.dropna(subset=["Source_ID", "Target_ID"])
        dep[["Source_ID", "Target_ID"]] = dep[["Source_ID", "Target_ID"]].astype(int)

        self.df_dep = dep.reset_index(drop=True)


    def _create_node_features(self) -> None:
        files = self.df["File"].astype(str).str.replace("\\", "/", regex=False).tolist()
        segs = {f: [s for s in "/".join(f.split("/")[:-1]).split("/") if s] for f in files}

        firsts = [ss[0] for ss in segs.values() if ss]
        common_root = firsts[0] if firsts and all(x == firsts[0] for x in firsts) else None

        drop = {"src", "main", "java"}
        drop.add(common_root) 

        def folder_tokens(f: str) -> list[str]:
            out, seen = [], set()
            for s in segs.get(f, []):
                if s not in drop and s not in seen:
                    out.append(s); seen.add(s)
            if out:
                return out

            name = f.split("/")[-1]
            base, ext = os.path.splitext(name)
            toks = re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+", base.lower())
            if ext.lower() in {".c", ".h", ".cpp", ".hpp"}:
                return (["root"] if "/" not in f else [f.split("/")[0]]) + toks
            return toks


        loc_texts = [" ".join(folder_tokens(f)) or "root" for f in files]
        self.df["Loc_features"] = loc_texts
        loc_features = CountVectorizer(binary=True).fit_transform(loc_texts).toarray().astype(np.float32)

        base = self.df[["Entity", "Code"]].copy()
        base["Entity"] = base["Entity"].astype(str)
        base["Code"] = base["Code"].fillna("").astype(str)
        emb_map = W2VEmbeddingGenerator(base, max_df=self.max_df).generate(**self.w2v_params)

        if emb_map:
            self._w2v_dim = next(iter(emb_map.values())).shape[0]
            zero = np.zeros(self._w2v_dim, dtype=np.float32)

            code_features = np.vstack([emb_map.get(e, zero) for e in self.df["Entity"].astype(str)]).astype(np.float32)
            code_features = normalize(code_features, norm="l2").astype(np.float32)

            features = np.hstack([loc_features, code_features]).astype(np.float32)
        else:
            self._w2v_dim = 0
            features = loc_features

        self["entity"].x = torch.tensor(features, dtype=torch.float)
        self.nodes = self.df["File"]

        self.label_encoder = LabelEncoder().fit(self.df["Module"].astype(str))
        self.num_classes = len(self.label_encoder.classes_)
        self.df["Label"] = self.label_encoder.transform(self.df["Module"].astype(str))


    def _create_edges(self) -> None:
        if self.df_dep.empty:
            self.relations = []
            return

        self.relations = sorted(self.df_dep["Dependency_Type"].dropna().unique().tolist())
        for dep_type, g in self.df_dep.groupby("Dependency_Type"):
            src = torch.tensor(g["Source_ID"].to_numpy(dtype=np.int64), dtype=torch.long)
            tgt = torch.tensor(g["Target_ID"].to_numpy(dtype=np.int64), dtype=torch.long)
            self["entity", str(dep_type), "entity"].edge_index = torch.stack([src, tgt], dim=0)
