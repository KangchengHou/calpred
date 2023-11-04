#!/usr/bin/env python
from . import logger
import fire
import numpy as np
import pandas as pd
from typing import List, Union, Dict
from .utils import compute_group_stats


def log_params(name, params):
    logger.info(
        f"Received parameters: \n{name}\n  "
        + "\n  ".join(f"--{k}={v}" for k, v in params.items())
    )


def cli():
    """
    Entry point for the admix command line interface.
    """
    fire.Fire()


def group_stats(
    df: Union[pd.DataFrame, str],
    y: str,
    pred: str,
    group: Union[str, List[str]],
    out_prefix: str,
    predstd: str = None,
    cor: str = "spearman",
    n_subgroup: int = 5,
    n_bootstrap: int = 100,
    seed=1234,
):
    """
    Calculate the difference between the r2 between `y` and `pred`
    across groups of individuals. For each `group`, only rows with [y, pred, group] all
    present are used.

    Parameters
    ----------
    df : str
        Path to the dataframe containing the data.
    y : str
        Name of the column containing the observed phenotype.
    pred : str
        Name of the column containing the predicted value.
    group : Union[str, List[str]]
        Name of the column containing the group variable.
    out : str
        output prefix, <out>.r2.tsv, <out>.r2diff.tsv will be created
    predstd : str
        Name of the column containing predicted standard deviation.
    cor : str
        Correlation method to use. Options: spearman (default) or pearson.
    n_bootstrap : int
        Number of bootstraps to perform, default 1000.
    """

    np.random.seed(seed)
    # log_params("group-stats", locals())
    if isinstance(df, str):
        df = pd.read_csv(df, sep="\t", index_col=0)
    else:
        assert isinstance(df, pd.DataFrame), "df must be a str or a pd.DataFrame"
        df = df.copy()

    n_raw = df.shape[0]
    df.dropna(subset=[y, pred], inplace=True)
    logger.info(
        f"{df.shape[0]}/{n_raw} rows remains without missing values at {y} and {pred}"
    )
    if isinstance(group, str):
        group = [group]

    r2_df = []
    diff_df = []
    cat_df = []
    predint_df = []

    for col in group:
        # drop the entire row if one of col, y, pred is missing
        subset_cols = [col, y, pred] if predstd is None else [col, y, pred, predstd]
        tmp_df = df[subset_cols].dropna()
        unique_values = np.unique(tmp_df[col].values)
        n_unique = len(unique_values)
        if n_unique > n_subgroup:
            logger.info(f"Converting column '{col}' to {n_subgroup} subgroups")
            cat_var = pd.qcut(tmp_df[col], q=n_subgroup, duplicates="drop")
            df_col_cat = pd.DataFrame(
                enumerate(cat_var.cat.categories), columns=["q", "cat"]
            )
            df_col_cat.insert(0, "group", col)
            cat_df.append(df_col_cat)

            tmp_df[col] = cat_var.cat.codes
        else:
            logger.info(f"Column '{col}' has {n_unique} unique values: {unique_values}")

        res_df, res_se_df, r2_diff = compute_group_stats(
            tmp_df,
            y_col=y,
            pred_col=pred,
            predstd_col=predstd,
            group_col=col,
            n_bootstrap=n_bootstrap,
            cor=cor,
            return_r2_diff=True,
        )
        if predstd is not None:
            predint_df.append(
                pd.DataFrame(
                    {
                        "group": col,
                        "subgroup": res_df.index.values,
                        "coverage": res_df["coverage"].values,
                        "coverage_se": res_se_df["coverage"].values,
                        "length": res_df["length"].values,
                        "length_se": res_se_df["length"].values,
                    }
                )
            )
        r2_df.append(
            pd.DataFrame(
                {
                    "group": col,
                    "subgroup": res_df.index.values,
                    "r2": res_df["r2"].values,
                    "r2_se": res_se_df["r2"].values,
                }
            )
        )

        diff_df.append(
            [
                col,
                res_df["r2"].iloc[-1] - res_df["r2"].iloc[0],
                np.mean(r2_diff > 0),
                np.mean(r2_diff) / np.std(r2_diff),
            ]
        )

    ##############
    ### output ###
    ##############
    pd.concat(r2_df).to_csv(
        out_prefix + ".r2.tsv", sep="\t", index=False, float_format="%.6g"
    )
    pd.DataFrame(diff_df, columns=["group", "r2diff", "prob>0", "zscore"]).to_csv(
        out_prefix + ".r2diff.tsv",
        sep="\t",
        index=False,
        float_format="%.6g",
        na_rep="NA",
    )
    if len(cat_df) > 0:
        pd.concat(cat_df).to_csv(out_prefix + ".cat.tsv", sep="\t", index=False)

    if len(predint_df) > 0:
        pd.concat(predint_df).to_csv(
            out_prefix + ".predint.tsv", sep="\t", index=False, float_format="%.6g"
        )
