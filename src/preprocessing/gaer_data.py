from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple
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
        if "Entity" not in self.df.columns:
            self.df["Entity"] = self.df["File"].astype(str).str.replace("/", ".", regex=False)

        if "Code" not in self.df.columns:
            if "Member_Name" in self.df.columns:
                code_map = (
                    self.df.groupby("File")["Member_Name"]
                    .apply(lambda s: " ".join(sorted(set(str(x) for x in s if pd.notna(x) and str(x).strip()))))
                    .to_dict()
                )
                self.df["Code"] = self.df["File"].map(code_map).fillna("")
            else:
                self.df["Code"] = ""

        keep_cols = [c for c in ["Entity", "Module", "Duplicated", "Code"] if c in self.df.columns]
        agg = (
            self.df.groupby(["File"], as_index=False)
            .agg({
                **{k: "first" for k in keep_cols},
                "Module": lambda s: sorted(set([x for x in s if pd.notna(x)]))
            })
        )

        empty = agg["Module"].str.len() == 0
        if empty.any():
            agg.loc[empty, "Module"] = [["__none__"]] * int(empty.sum())

        agg["Duplicated"] = agg["Module"].str.len().gt(1)
        agg["Module_List"] = agg["Module"]
        agg["Module"] = agg["Module"].apply(lambda xs: xs[0])

        self.df = agg.dropna(subset=["File"]).drop_duplicates(subset=["File"]).reset_index(drop=True)
        valid_files = set(self.df["File"].astype(str))
        dep = self.df_dep.copy()

        if "Dependency_Count" not in dep.columns:
            dep["Dependency_Count"] = 1.0
        if "Dependency_Type" not in dep.columns:
            dep["Dependency_Type"] = "__dep__"

        dep["Source_File"] = dep["Source_File"].astype(str)
        dep["Target_File"] = dep["Target_File"].astype(str)
        dep["Dependency_Type"] = dep["Dependency_Type"].fillna("__dep__").astype(str)
        dep["Dependency_Count"] = pd.to_numeric(dep["Dependency_Count"], errors="coerce").fillna(1.0)

        dep = dep[dep["Source_File"].isin(valid_files) & dep["Target_File"].isin(valid_files)]
        dep = dep[~dep["Dependency_Type"].str.contains("possible", case=False, na=False)]

        dep = dep.groupby(["Source_File", "Target_File", "Dependency_Type"], as_index=False)["Dependency_Count"].sum()

        self.df["File_ID"] = range(len(self.df))
        idx_map = dict(zip(self.df["File"].astype(str), self.df["File_ID"].astype(int)))

        dep["Source_ID"] = dep["Source_File"].map(idx_map)
        dep["Target_ID"] = dep["Target_File"].map(idx_map)
        dep = dep.dropna(subset=["Source_ID", "Target_ID"])
        dep[["Source_ID", "Target_ID"]] = dep[["Source_ID", "Target_ID"]].astype(int)

        self.df_dep = dep.reset_index(drop=True)

    def _create_node_features(self) -> None:
        base = self.df[["File", "Entity", "Code"]].copy()
        base["File"] = base["File"].astype(str)
        base["Entity"] = base["Entity"].astype(str)
        base["Code"] = base["Code"].fillna("").astype(str)

        gen = W2VEmbeddingGenerator(base[["Entity", "Code"]], max_df=self.max_df)
        emb_map = gen.generate(**self.w2v_params)

        if emb_map:
            self._w2v_dim = next(iter(emb_map.values())).shape[0]
            zero = np.zeros(self._w2v_dim, dtype=np.float32)

            file2ents = base.groupby("File")["Entity"].apply(list).to_dict()
            self._file_w2v = {}
            for f, ents in file2ents.items():
                vecs = [emb_map.get(e, zero) for e in ents if e in emb_map]
                self._file_w2v[f] = np.mean(vecs, axis=0).astype(np.float32) if vecs else zero
        else:
            self._file_w2v, self._w2v_dim = {}, 0

        def trim_without_ext(path: str, lang: str) -> str:
            s = str(path).replace("\\", "/").lower()
            lang = (lang or "java").lower()
            s_noext = s.rsplit(".", 1)[0] if "." in s else s

            if lang == "java":
                dir_part = s_noext.rsplit("/", 1)[0] if "/" in s_noext else ""
                toks = [t for t in dir_part.split("/") if t]
                return " ".join(toks)

            if lang in ("c", "cpp", "c++"):
                toks = [t for t in re.split(r"[/\.]+", s_noext) if t]
                return " ".join(toks)

            dir_part = s_noext.rsplit("/", 1)[0] if "/" in s_noext else ""
            toks = [t for t in dir_part.split("/") if t]
            return " ".join(toks)

        lang = infer_language_from_files(self.df["File"])
        base_names = self.df["File"].astype(str).map(lambda s: trim_without_ext(s, lang))

        loc_vectorizer = CountVectorizer(binary=True)
        loc_features = loc_vectorizer.fit_transform(base_names).toarray()
        if self._w2v_dim > 0 and self._file_w2v:
            zero = np.zeros(self._w2v_dim, dtype=np.float32)
            code_features = np.vstack([self._file_w2v.get(f, zero) for f in self.df["File"].astype(str)])
            code_features = normalize(code_features, norm="l2")
            features = np.hstack([loc_features, code_features])
        else:
            features = loc_features

        self["entity"].x = torch.tensor(features, dtype=torch.float)
        self.nodes = self.df["File"]

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
