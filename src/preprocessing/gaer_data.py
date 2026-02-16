from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

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


def strip_extension(path: str) -> str:
    s = str(path)
    i = s.rfind(".")
    return s[:i] if i > 0 else s


class HeterogeneousData(HeteroData):
    def __init__(
        self,
        df: pd.DataFrame,
        df_dep: pd.DataFrame,
        language: Optional[str] = None,
        w2v_params: Optional[dict] = None,
        max_df: float = 0.9,
    ):
        super().__init__()
        self.df = df.copy()
        self.df_dep = df_dep.copy()

        self.language = (language or infer_language_from_files(self.df.get("File", pd.Series(dtype=str)))).lower()
        self.w2v_params = w2v_params or dict(
            vector_size=100, window=5, min_count=5, sg=1, epochs=10, max_vocab_size=2000
        )
        self.max_df = float(max_df)

        self._file_w2v: Dict[str, np.ndarray] = {}
        self._w2v_dim: int = 0
        self.label_encoder: Optional[LabelEncoder] = None
        self.num_classes: Optional[int] = None
        self.relations: List[str] = []
        self.has_multilabels: bool = False

        self._clean_tables()
        self._create_node_features()
        self._create_edges()

    def _clean_tables(self) -> None:
        if "File" not in self.df.columns:
            raise ValueError("df must contain a 'File' column.")

        nodes = self.df.copy()
        nodes["File"] = nodes["File"].astype(str)

        if "Member_Name" in nodes.columns:
            nodes["Member_Name"] = nodes["Member_Name"].fillna("").astype(str).str.strip()
            nodes = nodes[nodes["Member_Name"] != ""]
            code_df = (
                nodes.groupby("File", as_index=False)["Member_Name"]
                .apply(lambda s: " ".join(s.tolist()))
                .rename(columns={"Member_Name": "Code"})
            )
        else:
            code_df = pd.DataFrame({"File": nodes["File"].drop_duplicates(), "Code": ""})

        if "Module" in nodes.columns:
            nodes["Module"] = nodes["Module"].fillna("").astype(str)
            mod_df = (
                nodes.groupby("File", as_index=False)["Module"]
                .apply(lambda s: sorted({m for m in s.tolist() if m and m.lower() != "nan"}))
                .rename(columns={"Module": "Module_List"})
            )
        else:
            mod_df = pd.DataFrame({"File": nodes["File"].drop_duplicates(), "Module_List": [["__none__"]]})

        out = nodes[["File"]].drop_duplicates()
        out = pd.merge(out, mod_df, on="File", how="left")
        out = pd.merge(out, code_df, on="File", how="left")

        out["Module_List"] = out["Module_List"].apply(lambda xs: xs if xs else ["__none__"])
        out["Duplicated"] = out["Module_List"].str.len().gt(1)
        out["Module"] = out["Module_List"].apply(lambda xs: xs[0] if xs else "__none__")
        out["Entity"] = out["File"]
        out["File_ID"] = range(len(out))

        self.df = out.reset_index(drop=True)
        self.has_multilabels = bool(self.df["Duplicated"].any())

        dep = self.df_dep.copy()
        if dep is None or dep.empty:
            self.df_dep = pd.DataFrame(columns=["Source_File", "Target_File", "Source_ID", "Target_ID", "Dependency_Type"])
            return

        if "Source_File" not in dep.columns or "Target_File" not in dep.columns:
            raise ValueError("df_dep must contain 'Source_File' and 'Target_File' columns.")

        dep["Source_File"] = dep["Source_File"].astype(str)
        dep["Target_File"] = dep["Target_File"].astype(str)
        dep = dep[dep["Source_File"] != dep["Target_File"]]
        dep = dep.drop_duplicates(subset=["Source_File", "Target_File"]).reset_index(drop=True)

        if "Dependency_Type" not in dep.columns:
            dep["Dependency_Type"] = "__dep__"
        dep["Dependency_Type"] = dep["Dependency_Type"].fillna("__dep__").astype(str)
        dep = dep[~dep["Dependency_Type"].str.contains("possible", case=False, na=False)]

        valid = set(self.df["File"].astype(str))
        dep = dep[dep["Source_File"].isin(valid) & dep["Target_File"].isin(valid)].reset_index(drop=True)

        idx = dict(zip(self.df["File"].astype(str), self.df["File_ID"].astype(int)))
        dep["Source_ID"] = dep["Source_File"].map(idx)
        dep["Target_ID"] = dep["Target_File"].map(idx)
        dep = dep.dropna(subset=["Source_ID", "Target_ID"])
        dep[["Source_ID", "Target_ID"]] = dep[["Source_ID", "Target_ID"]].astype(int)

        self.df_dep = dep.reset_index(drop=True)


    def _create_node_features(self) -> None:
        gen = W2VEmbeddingGenerator(self.df[["Entity", "Code"]], max_df=self.max_df)
        emb_map = gen.generate(**self.w2v_params)

        if emb_map:
            self._w2v_dim = next(iter(emb_map.values())).shape[0]
            zero = np.zeros(self._w2v_dim, dtype=np.float32)
            self._file_w2v = {f: emb_map.get(f, zero) for f in self.df["File"].astype(str)}
        else:
            self._w2v_dim = 0
            self._file_w2v = {}

        nodes_names = self.df["File"].astype(str).map(strip_extension)
        loc_vec = CountVectorizer(binary=False)
        loc_features = loc_vec.fit_transform(nodes_names).toarray()

        if self._w2v_dim > 0:
            zero = np.zeros(self._w2v_dim, dtype=np.float32)
            code_features = np.vstack([self._file_w2v.get(f, zero) for f in self.df["File"].astype(str)])
            code_features = normalize(code_features, norm="l2")
            features = np.hstack([loc_features, code_features])
        else:
            features = loc_features

        self["entity"].x = torch.tensor(features, dtype=torch.float)
        self.nodes = self.df.File
        all_mods = sorted({m for xs in self.df["Module_List"] for m in (xs or [])}) or ["__none__"]
        self.label_encoder = LabelEncoder().fit(all_mods)
        self.num_classes = len(self.label_encoder.classes_)
        self.df["Label"] = self.label_encoder.transform(self.df["Module"].astype(str))

    def _create_edges(self) -> None:
        if self.df_dep.empty:
            self.relations = []
            return
        self.relations = sorted(self.df_dep["Dependency_Type"].dropna().unique().tolist())
        edge_dict: Dict[str, List[Tuple[int, int]]] = defaultdict(list)

        for _, row in self.df_dep.iterrows():
            edge_dict[row["Dependency_Type"]].append((int(row["Source_ID"]), int(row["Target_ID"])))

        for dep_type, edges in edge_dict.items():
            if not edges:
                continue
            src, tgt = zip(*edges)
            self["entity", dep_type, "entity"].edge_index = torch.tensor([src, tgt], dtype=torch.long)
