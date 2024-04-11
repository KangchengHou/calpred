# CalPred (Calibrated Prediction Intervals for Polygenic Scores Across Diverse Contexts)

See [companion manuscript github repository](https://github.com/KangchengHou/calpred-manuscript) for analysis scripts used in the [manuscript](https://www.medrxiv.org/content/10.1101/2023.07.24.23293056v1).

## Installation
```bash
# calpred calls R packages statmod and Rchoice in fitting the model
Rscript -e "install.packages(c('statmod', 'Rchoice'), repos='https://cran.rstudio.com')"
pip install calpred
```

## Quick example
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import calpred

np.random.seed(42)
n = 1000
pgs = np.random.normal(size=n)
age = np.random.normal(loc=40, scale=10, size=n)
sex = np.random.binomial(n=1, p=0.5, size=n)

y_mean = 8 + pgs * 0.5 + age * -0.2 + sex * 0.5
y_sd = np.sqrt(np.exp(2 + age * -0.03 + sex * 1))
y = np.random.normal(loc=y_mean, scale=y_sd)

df = pd.DataFrame({"intercept": 1, "pgs": pgs, "age": age, "sex": sex, "y": y})

# x and z are the columns for fitting the mean and standard deviation
x = z = df[["intercept", "pgs", "age", "sex"]]
model = calpred.fit(y=df["y"], x=x, z=z)

# prediction mean and [low, high] for 90% prediction interval
df["pred_mean"], df["pred_sd"] = calpred.predict(x=x, z=z, model_fit=model)
df["pred_low"] = df["pred_mean"] - df["pred_sd"] * 1.645
df["pred_high"] = df["pred_mean"] + df["pred_sd"] * 1.645


# show prediction intervals at 5% / 95% quantile of prediction mean
fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
ax.scatter(df["pred_mean"], df["y"], s=4)
ax.axline((0, 0), slope=1, ls="--", color="red")

idx1 = df.sort_values("pred_mean").index[int(n * 0.05)]
idx2 = df.sort_values("pred_mean").index[int(n * 0.95)]

for idx in [idx1, idx2]:
    ax.errorbar(
        x=df.loc[idx, "pred_mean"],
        y=df.loc[idx, "pred_mean"],
        yerr=df.loc[idx, "pred_sd"] * 1.645,
        color="red",
        capsize=3,
        lw=1,
    )
fig.show()
```

## Upload to PyPI (for developers)
```bash
python setup.py sdist
twine upload dist/*
```
