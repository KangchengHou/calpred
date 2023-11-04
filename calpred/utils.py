from . import logger
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List


def quantile_normalize(values: np.ndarray):
    from scipy.stats import rankdata, norm

    values = np.array(values)
    non_nan_index = ~np.isnan(values)
    results = np.full(values.shape, np.nan)
    results[non_nan_index] = norm.ppf(
        (rankdata(values[non_nan_index]) - 0.5) / len(values[non_nan_index])
    )
    return results


@dataclass
class QMap:
    """Map real values to quantiles"""

    q2x = None
    x2q = None

    def __init__(self, values):
        from scipy.interpolate import interp1d

        values = values[~np.isnan(values)]
        q = quantile_normalize(values)
        self.x2q = interp1d(values, q, fill_value=(min(q), max(q)), bounds_error=False)
        self.q2x = interp1d(
            q, values, fill_value=(min(values), max(values)), bounds_error=False
        )


def compute_group_stats(
    df: pd.DataFrame,
    y_col: str,
    pred_col: str,
    predstd_col: str = None,
    ci=0.90,
    group_col: str = None,
    n_bootstrap: int = 0,
    cor: str = "pearson",
    return_r2_diff=False,
):
    """
    Summarize the results of prediction:
    with R2, coverage, interval length for different groups.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the data, must have columns `y_col`, `pred_col`
    y_col : str
        Name of the column containing the true value.
    pred_col : str
        predicted value
    predstd_col : str
        standard deviation of the prediction
    group_col : str
        Name of the column containing the group variable.
    cor : str
        correlation method, default pearson, can be pearson, spearman
    return_r2_diff : bool
        whether to return standard error of prediction,

    Returns
    -------
    pandas.DataFrame
        Dataframe with n_level rows and three columns `group_col`, `r2`, `coverage`,
        and `interval_length`
    """
    val_cols = [y_col, pred_col]
    if group_col is not None:
        val_cols.append(group_col)
    if predstd_col is not None:
        val_cols.append(predstd_col)

    # dropping rows with missing values
    df = df[val_cols].dropna()

    # cor_func check whether only 1 value is present.
    if cor == "pearson":
        cor_func = lambda x, y: stats.pearsonr(x, y)[0] if len(x) > 1 else np.nan
    elif cor == "spearman":
        cor_func = lambda x, y: stats.spearmanr(x, y)[0] if len(x) > 1 else np.nan
    else:
        raise ValueError(f"cor must be pearson or spearman, got {cor}")

    ci_z = stats.norm.ppf((1 + ci) / 2)
    if group_col is not None:
        df_grouped = df.groupby(group_col)
        r2 = df_grouped.apply(lambda df: cor_func(df[pred_col], df[y_col]) ** 2)
        y_std = df_grouped.apply(lambda df: df[y_col].std())
        pred_std = df_grouped.apply(lambda df: df[pred_col].std())
        res_df = {
            "r2": r2,
            "std(y)": y_std,
            "std(pred)": pred_std,
        }

        if predstd_col is not None:
            res_df["coverage"] = df_grouped.apply(
                lambda df: df[y_col]
                .between(
                    df[pred_col] - df[predstd_col] * ci_z,
                    df[pred_col] + df[predstd_col] * ci_z,
                )
                .mean()
            )
            res_df["length"] = df_grouped.apply(
                lambda df: (df[predstd_col] * ci_z).mean()
            )
        res_df = pd.DataFrame(res_df)
    else:
        r2 = cor_func(df[pred_col], df[y_col]) ** 2
        y_std = df[y_col].std()
        pred_std = df[pred_col].std()

        res_df = {
            "r2": r2,
            "std(y)": y_std,
            "std(pred)": pred_std,
        }
        if predstd_col is not None:
            res_df["coverage"] = (
                df[y_col]
                .between(
                    df[pred_col] - df[predstd_col] * ci_z,
                    df[pred_col] + df[predstd_col] * ci_z,
                )
                .mean()
            )
            res_df["length"] = (df[predstd_col] * ci_z).mean()
        res_df = pd.Series(res_df)
    if n_bootstrap == 0:
        return res_df
    else:
        bootstrap_dfs = []
        for _ in tqdm(range(n_bootstrap), desc="Bootstrapping"):
            # sample with replacement
            bootstrap_dfs.append(
                compute_group_stats(
                    df.sample(frac=1, replace=True),
                    y_col,
                    pred_col,
                    predstd_col,
                    ci,
                    group_col,
                    n_bootstrap=0,
                )
            )
        if isinstance(res_df, pd.Series):
            res_se_df = pd.Series(
                np.dstack(bootstrap_dfs).std(axis=2).flatten(), index=res_df.index
            )
        else:
            # make sure bootstrap dataframe has the same index with the original dataframe
            bootstrap_dfs = [d.reindex(res_df.index) for d in bootstrap_dfs]
            res_se_df = pd.DataFrame(
                np.dstack(bootstrap_dfs).std(axis=2),
                index=res_df.index,  # type: ignore
                columns=res_df.columns,  # type: ignore
            )
        if (group_col is not None) and return_r2_diff:
            r2_diff = np.array(
                [d["r2"].iloc[-1] - d["r2"].iloc[0] for d in bootstrap_dfs]
            )
            return res_df, res_se_df, r2_diff
        else:
            return res_df, res_se_df
