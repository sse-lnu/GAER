from __future__ import annotations

from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd


class NEGARData:
    def __init__(self, df: pd.DataFrame, df_dep: pd.DataFrame, use_majority_vote: bool = True):
        self.df = df.copy()
        self.df_dep = df_dep.copy()
        self.use_majority_vote = bool(use_majority_vote)

        self.file_to_label: pd.Series = pd.Series(dtype=str)
        self.G: nx.Graph = nx.Graph()
        self.node_list: list[str] = []
        self.nodes: pd.Series = pd.Series(dtype=str)
        self.y_true: Optional[np.ndarray] = None
        self.num_classes = df.Module.nunique()

        self._build_data()

    def _build_data(self) -> None:
        if "File" not in self.df.columns:
            raise ValueError("df must contain a 'File' column.")

        d = self.df
        d["File"] = d["File"].astype(str).str.strip()
        d["Module"] = d["Module"].astype(str).str.strip() if "Module" in d.columns else "__none__"

        if self.use_majority_vote:
            c = d.groupby(["File", "Module"]).size().reset_index(name="cnt")
            c = c.sort_values(["File", "cnt", "Module"], ascending=[True, False, True])
            nodes_df = c.drop_duplicates("File", keep="first")[["File", "Module"]]
        else:
            nodes_df = d[["File", "Module"]].drop_duplicates("File")

        self.df = nodes_df.reset_index(drop=True)
        self.file_to_label = self.df.set_index("File")["Module"]
        keep = set(self.file_to_label.index.tolist())

        dep = self.df_dep
        if dep is None or dep.empty:
            dep = pd.DataFrame(columns=["Source_File", "Target_File"])

        if "Source_File" not in dep.columns or "Target_File" not in dep.columns:
            raise ValueError("df_dep must contain 'Source_File' and 'Target_File' columns.")

        dep["Source_File"] = dep["Source_File"].astype(str).str.strip()
        dep["Target_File"] = dep["Target_File"].astype(str).str.strip()
        dep = dep[dep["Source_File"] != dep["Target_File"]]
        dep = dep.drop_duplicates(subset=["Source_File", "Target_File"]).reset_index(drop=True)
        dep = dep[dep["Source_File"].isin(keep) & dep["Target_File"].isin(keep)].reset_index(drop=True)

        self.df_dep = dep

        G = nx.Graph()
        G.add_nodes_from(sorted(keep))
        if not dep.empty:
            G.add_edges_from(dep[["Source_File", "Target_File"]].itertuples(index=False, name=None))

        self.G = G
        self.node_list = list(G.nodes())
        self.nodes = pd.Series(self.node_list, name="File")

        y_str = self.file_to_label.loc[self.node_list].astype(str).tolist()
        y_true, _ = pd.factorize(y_str)
        self.y_true = y_true.astype(int)
