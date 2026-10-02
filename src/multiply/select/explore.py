import numpy as np
import pandas as pd

from .multiplex import MultiplexResults


class MultiplexExplorer:
    def __init__(self, primer_df, multiplexes):
        """
        Explore `multiplexes` derived from a given `primer_df`.

        params
            primer_df: pandas DataFrame
                Each row is an individual primer.
            multiplexes: List[Multiplex] | MultiplexResults
                Either the legacy fully-materialised list of Multiplex
                objects (e.g. from :class:`BruteForce`), or a
                :class:`MultiplexResults` container holding the raw
                ``(N, n_targets)`` index array + per-row costs from a
                jitted selector. The lazy path defers per-Multiplex
                Python construction past the dedup/sort and pays it
                only for the ``top_N`` picks materialised in
                :py:meth:`set_top_multiplexes`.
        """

        self.primer_df = primer_df
        self.multiplexes = multiplexes

        if isinstance(multiplexes, MultiplexResults):
            self._init_from_results(multiplexes)
        else:
            self._init_from_list(multiplexes)

    def _init_from_list(self, multiplexes):
        self._results = None
        self.uniq_multiplexes = sorted(set(multiplexes), key=lambda m: m.cost)
        self.uniq_costs = np.array([m.cost for m in self.uniq_multiplexes], dtype=np.float64)
        self.N_uniq = len(self.uniq_multiplexes)

    def _init_from_results(self, results: "MultiplexResults"):
        """Dedup rows of the index array, then sort the uniques by cost.

        Dedup keys on the row bytes — multiplexes are unique iff the
        per-target index vector matches exactly. Costs alone don't
        suffice because two distinct picks can collide on cost.
        """
        self._results = results
        all_idx = results.all_indices
        all_costs = results.all_costs

        seen: dict[bytes, int] = {}
        keep: list[int] = []
        for it in range(all_idx.shape[0]):
            key = all_idx[it].tobytes()
            if key not in seen:
                seen[key] = it
                keep.append(it)
        keep_arr = np.asarray(keep, dtype=np.int64)
        order = np.argsort(all_costs[keep_arr])
        self._uniq_indices = keep_arr[order]
        self.uniq_costs = all_costs[self._uniq_indices].copy()
        self.N_uniq = self._uniq_indices.shape[0]
        # Lazy: don't build full Multiplex list for all uniques. Consumers
        # iterating over costs should use ``self.uniq_costs`` directly.
        # Anything needing the full list (e.g. ``[m.cost for m in ...]``)
        # still works via materialisation, but at top-N granularity.
        self.uniq_multiplexes = None  # set on demand via set_top_multiplexes

    def _extract_from_df(self, df):
        """
        Extract multiplex rows from a dataframe

        """

        edf = df.copy()
        edf.index = df["primer_name"]

        # Extract associated dataframes
        multiplex_dfs = [edf.loc[m.get_primer_names()] for m in self.top_multiplexes]

        return multiplex_dfs

    def set_top_multiplexes(self, top_N=3):
        """Materialise the ``top_N`` lowest-cost unique multiplexes.

        For the lazy (MultiplexResults) path this is where Python-side
        Multiplex objects first get built — only ``top_N`` of them,
        not N_uniq.
        """
        if top_N > self.N_uniq:
            print(f"  Requested top {top_N} multiplexes, but only {self.N_uniq} multiplexes found.")
            print("  **Consider re-running with brute force selection algorithm.**")
            print(f"  Will return {self.N_uniq} multiplexes.")
            self.top_N = self.N_uniq
        else:
            self.top_N = top_N

        if self._results is None:
            self.top_multiplexes = self.uniq_multiplexes[:self.top_N]
        else:
            self.top_multiplexes = self._results.materialise(
                self._uniq_indices[:self.top_N].tolist()
            )

    def get_union_dataframe(self, output_path=None):
        """
        Write a union dataframe giving information about union of
        primers across all multiplexes

        """

        # Extract multiplex dataframes
        dfs = self._extract_from_df(df=self.primer_df)

        # Combine
        union_df = (
            pd.concat(dfs)
            .reset_index(drop=True)
            .fillna(False)
            .drop_duplicates("primer_name")
            .sort_values("primer_name")
        )

        # Add inclusion column. `primer_pairs` is a tuple so `in` is O(K) per
        # row; precompute as a set for O(1) membership.
        for ix in range(self.top_N):
            pairs_set = set(self.top_multiplexes[ix].primer_pairs)
            in_multiplex = [p in pairs_set for p in union_df["pair_name"]]
            union_df.insert(ix + 4, f"in_multiplex{ix:02d}", in_multiplex)

        # Store
        self.union_df = union_df

        # Optionally write
        if output_path is not None:
            self.union_df.to_csv(output_path, index=False)

    def get_order_dataframe(self, output_path=None):
        """Write dataframe for ordering"""
        assert self.union_df is not None, "Must run `.get_union_dataframe()` first."

        # Create smaller datafram with columns for IDT order of primers
        order_df = self.union_df[["primer_name", "seq"]]
        order_df.insert(2, "concentration", "25nm")
        order_df.insert(3, "purification", "STD")

        # Store
        self.order_df = order_df

        if output_path is not None:
            self.order_df.to_csv(output_path, index=False)
