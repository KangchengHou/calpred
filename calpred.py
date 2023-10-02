import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
from typing import List, Union
import structlog
import fire

logger = structlog.get_logger()


def log_params(name, params):
    logger.info(
        f"Received parameters: \n{name}\n  "
        + "\n  ".join(f"--{k}={v}" for k, v in params.items())
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


if __name__ == "__main__":
    fire.Fire()
