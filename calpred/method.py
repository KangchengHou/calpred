import numpy as np
import pandas as pd
import statsmodels.api as sm
import subprocess
import tempfile
import os
from dataclasses import dataclass
import json
from scipy import stats


@dataclass
class CalPredFit:
    """A dataclass for representing calibration prediction fits.

    Parameters

    Attributes:
        mean_coef (pd.Series): mean coefficients.
        mean_se (pd.Series): standard errors of the mean coefficients.
        sd_coef (pd.Series): standard deviation coefficients.
        sd_se (pd.Series): standard errors of the standard deviation.

    Example:
        # Create pandas Series
        mean_coef_series = pd.Series([1.0, 2.0, 3.0], name="mean_coef")
        mean_se_series = pd.Series([0.1, 0.2, 0.3], name="mean_se")
        sd_coef_series = pd.Series([2.0, 4.0, 6.0], name="sd_coef")
        sd_se_series = pd.Series([0.2, 0.4, 0.6], name="sd_se")

        # Create CalPredFit instance
        model = CalPredFit(
            mean_coef=mean_coef_series,
            mean_se=mean_se_series,
            sd_coef=sd_coef_series,
            sd_se=sd_se_series
        )

        # Save to JSON
        model.to_json('cal_pred_fit.json')

        # Load from JSON
        model = CalPredFit.from_json('cal_pred_fit.json')
    """

    mean_coef: pd.Series
    mean_se: pd.Series
    sd_coef: pd.Series
    sd_se: pd.Series

    def __repr__(self):
        """Custom string representation for CalPredFit."""
        mean_str = pd.DataFrame(
            {
                "coef": self.mean_coef,
                "se": self.mean_se,
                "z-score": self.mean_coef / self.mean_se,
            }
        ).to_string()
        sd_str = pd.DataFrame(
            {
                "coef": self.sd_coef,
                "se": self.sd_se,
                "z-score": self.sd_coef / self.sd_se,
            }
        ).to_string()

        return (
            f"CalPredFit(\n"
            + "-" * 40
            + "\n"
            + "Estimates for the mean:\n"
            + mean_str
            + "\n"
            + "-" * 6
            + "\n"
            + "Estimates for the sd:\n"
            + sd_str
            + "\n"
            + "-" * 40
            + "\n"
            ")"
        )

    def to_json(self, path: str):
        """Serialize the instance to a JSON file.

        Parameters
        ----------
        path: str
            The name of the JSON file to save the data.
        """
        data_dict = {
            "mean_coef": self.mean_coef.to_dict(),
            "mean_se": self.mean_se.to_dict(),
            "sd_coef": self.sd_coef.to_dict(),
            "sd_se": self.sd_se.to_dict(),
        }

        with open(path, "w") as f:
            json.dump(data_dict, f, indent=2)

    @classmethod
    def from_json(self, path: str):
        """Deserialize the instance from a JSON file.

        Parameters
        ----------
        path: str
            The name of the JSON file to load the data.

        Returns
        -------
        CalPredFit: An instance of CalPredFit loaded from the JSON file.

        Example
        -------
        new_cal_pred_fit = CalPredFit.from_json('cal_pred_fit.json')
        """
        with open(path, "r") as f:
            data_dict = json.load(f)

        return self(
            mean_coef=pd.Series(data_dict["mean_coef"]),
            mean_se=pd.Series(data_dict["mean_se"]),
            sd_coef=pd.Series(data_dict["sd_coef"]),
            sd_se=pd.Series(data_dict["sd_se"]),
        )


def load_calpred_fit(path: str):
    """Load a CalPredFit from a JSON file.

    Parameters
    ----------
    path: str
        The name of the JSON file to load the data.

    Returns
    -------
    CalPredFit: An instance of CalPredFit loaded from the JSON file.
    """
    return CalPredFit.from_json(path)


def fit(y: np.ndarray, x: pd.DataFrame, z: pd.DataFrame, rscript_bin="Rscript"):
    """Fit CalPred model

    Parameters
    ----------
    y : np.ndarray
        response variable
    x : pd.DataFrame
        data matrix for mean effects, without intercept
    z : pd.DataFrame
        data matrix for standard errors, without intercept

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

    subprocess.run([rscript_bin, rscript, y_file, x_file, z_file, out_prefix])

    mean_coef = np.loadtxt(out_prefix + ".mean")
    sd_coef = np.loadtxt(out_prefix + ".sd")

    if z.shape[1] == 1:
        # make sd_coef a column vector
        sd_coef = sd_coef.reshape(1, -1)

    tmp_dir_obj.cleanup()

    return CalPredFit(
        pd.Series(mean_coef[:, 0], index=x.columns),
        pd.Series(mean_coef[:, 1], index=x.columns),
        pd.Series(sd_coef[:, 0], index=z.columns),
        pd.Series(sd_coef[:, 1], index=z.columns),
    )


def predict(x: pd.DataFrame, z: pd.DataFrame, model_fit: CalPredFit):
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
    x_coef = model_fit.mean_coef
    z_coef = model_fit.sd_coef
    assert np.all(x.index == z.index)
    x = sm.add_constant(x)
    z = sm.add_constant(z)

    assert np.all(x.columns == x_coef.index)
    assert np.all(z.columns == z_coef.index)

    y_mean = x.dot(x_coef)
    y_sd = np.sqrt(np.exp(z.dot(z_coef)))
    return y_mean, y_sd


def fit_binary(
    y: np.ndarray,
    x: pd.DataFrame,
    z: pd.DataFrame,
    verbose: bool = False,
    rscript_bin="Rscript",
):
    """Fit CalPred model for binary response variable

    Parameters
    ----------
    y : np.ndarray
        response variable
    x : pd.DataFrame
        data matrix for mean effects, without intercept
    z : pd.DataFrame
        data matrix for standard errors, without intercept
    verbose : bool
        whether to output intermediate model fitting information
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
    # z cannot have constant
    # z = sm.add_constant(z)

    tmp_dir_obj = tempfile.TemporaryDirectory()
    tmp_dir = tmp_dir_obj.name
    y_file = os.path.join(tmp_dir, "y.txt")
    x_file = os.path.join(tmp_dir, "x.txt")
    z_file = os.path.join(tmp_dir, "z.txt")
    np.savetxt(y_file, y)
    np.savetxt(x_file, x)
    np.savetxt(z_file, z)

    out_prefix = os.path.join(os.path.abspath(tmp_dir), "fit")
    rscript = os.path.join(os.path.dirname(__file__), "calpred_binary.R")

    subprocess.run([rscript_bin, rscript, y_file, x_file, z_file, out_prefix])

    mean_coef = np.loadtxt(out_prefix + ".mean")
    sd_coef = np.loadtxt(out_prefix + ".sd")

    if z.shape[1] == 1:
        # make sd_coef a column vector
        sd_coef = sd_coef.reshape(1, -1)

    tmp_dir_obj.cleanup()

    return CalPredFit(
        pd.Series(mean_coef[:, 0], index=x.columns),
        pd.Series(mean_coef[:, 1], index=x.columns),
        pd.Series(sd_coef[:, 0], index=z.columns),
        pd.Series(sd_coef[:, 1], index=z.columns),
    )


def predict_binary(x: pd.DataFrame, z: pd.DataFrame, model_fit: CalPredFit):
    """CalPred prediction for binary response varaibles

    Parameters
    ----------
    x : pd.DataFrame
        mean covariates
    z : pd.DataFrame
        sd covariates
    model_fit: CalpredFit
        model_fit
    """
    x_coef = model_fit.mean_coef
    z_coef = model_fit.sd_coef
    assert np.all(x.index == z.index)
    x = sm.add_constant(x)
    z = z

    assert np.all(x.columns == x_coef.index)
    assert np.all(z.columns == z_coef.index)

    y_mean = x.dot(x_coef)
    # sqrt(exp(z)) = exp(z/2)
    y_sd = np.sqrt(np.exp(z.dot(z_coef)))
    y_prob = stats.norm.cdf(y_mean / y_sd)

    return y_prob, y_mean, y_sd
