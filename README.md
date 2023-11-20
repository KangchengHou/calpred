# CalPred (Calibrated Prediction Intervals for Polygenic Scores Across Diverse Contexts)

See [companion manuscript github repository](https://github.com/KangchengHou/calpred-manuscript) for analysis scripts used in the [manuscript](https://www.medrxiv.org/content/10.1101/2023.07.24.23293056v1).

## Installation
```bash
git clone https://github.com/KangchengHou/calpred.git && cd calpred/
Rscript -e "install.packages(c('statmod', 'Rchoice', 'logger', 'optparse', 'glue'), repos='https://cran.rstudio.com')" # calpred dependency
Rscript -e "install.packages(c('devtools', 'ggplot2', 'dplyr', 'patchwork'), repos='https://cran.rstudio.com')" # for example notebooks
```

## Quick example
```bash
# df must have person ID as 1st column, and should not contain missing data
# y_col is the column name of the trait of interest
# mean_cols and sd_cols are columns fot fitting the mean and standard deviation
# with names separated by commas
# <out_prefix>.fitted.tsv (fitted mean and sd) and <out_prefix>.coef.tsv (coefficients) will be generated
# see toy/simulate.ipynb for the data simulation process
Rscript calpred.cli.R \
    --df toy/trait.tsv \
    --y_col pheno \
    --mean_cols pgs,ancestry,age,sex \
    --sd_cols pgs,ancestry,age,sex \
    --out_prefix toy/trait
```

## Example notebooks
- [Introduction with a simulated dataset](https://nbviewer.org/github/KangchengHou/calpred/blob/main/introduction.ipynb)
- [Example analysis pipeline](https://nbviewer.org/github/KangchengHou/calpred/blob/main/example.ipynb)

## Upload to PyPI
```bash
python setup.py sdist
twine upload dist/*
```
