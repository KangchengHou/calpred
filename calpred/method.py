import numpy as np
import pandas as pd
import statsmodels.api as sm
import subprocess
import tempfile
import os


def fit(y: np.ndarray, x: pd.DataFrame, z: pd.DataFrame):
    """Fit CalPred model

    Parameters
    ----------
    y : np.ndarray
        response variable
    x : pd.DataFrame
        data matrix for mean effects
    z : pd.DataFrame
        data matrix for standard errors

    Returns
    -------
    mean_coef : pd.Series
        mean effect coefficients
    sd_coef : pd.Series
        standard deviation coefficients
    mean_se : pd.Series
        standard error of mean effect coefficients
    sd_se : pd.Series
        standard error of standard deviation coefficients
    """

    x = sm.add_constant(x)
    z = sm.add_constant(z)

    tmp_dir_obj = tempfile.TemporaryDirectory()
    tmp_dir = tmp_dir_obj.name
    y_file = os.path.join(tmp_dir, "y.txt")
    x_file = os.path.join(tmp_dir, "x.txt")
    z_file = os.path.join(tmp_dir, "z.txt")
    np.savetxt(y_file, y)
    np.savetxt(x_file, x)
    np.savetxt(z_file, z)

    out_prefix = os.path.join(os.path.abspath(tmp_dir), "fit")
    rscript = os.path.join(os.path.dirname(__file__), "calpred.R")

    subprocess.run(["Rscript", rscript, y_file, x_file, z_file, out_prefix])

    mean_coef = np.loadtxt(out_prefix + ".mean")
    sd_coef = np.loadtxt(out_prefix + ".sd")

    tmp_dir_obj.cleanup()

    # mean_coef, sd_coef, mean_se, sd_se
    return (
        pd.Series(mean_coef[:, 0], index=x.columns),
        pd.Series(mean_coef[:, 1], index=x.columns),
        pd.Series(sd_coef[:, 0], index=z.columns),
        pd.Series(sd_coef[:, 1], index=z.columns),
    )


def predict(x: pd.DataFrame, z: pd.DataFrame, x_coef: pd.Series, z_coef: pd.Series):
    """predict CalPred model

    Parameters
    ----------
    x : pd.DataFrame
        mean covariates
    z : pd.DataFrame
        sd covariates
    x_coef : pd.Series
        mean coefficients
    z_coef : pd.Series
        standard deviation coefficients
    """
    assert np.all(x.index == z.index)
    x = sm.add_constant(x)
    z = sm.add_constant(z)

    assert np.all(x.columns == x_coef.index)
    assert np.all(z.columns == z_coef.index)

    y_mean = x.dot(x_coef)
    y_sd = np.sqrt(np.exp(z.dot(z_coef)))
    return y_mean, y_sd
