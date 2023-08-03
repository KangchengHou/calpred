# CalPred (Calibrated Prediction Intervals for Polygenic Scores Across Diverse Contexts)

> The package is still in early development. And therefore any comments/suggestions are highly appreciated. 

See [companion manuscript github repository](https://github.com/KangchengHou/calpred-manuscript) for analysis scripts used in the [manuscript](https://www.medrxiv.org/content/10.1101/2023.07.24.23293056v1).

## Try it out
```bash
git clone https://github.com/KangchengHou/calpred.git && cd calpred/
Rscript -e "install.packages('statmod', repos='https://cran.rstudio.com')" # calpred dependency
Rscript -e "install.packages(c('devtools', 'ggplot2', 'dplyr', 'patchwork'), repos='https://cran.rstudio.com')" # for example notebooks
# go to ./introduction.ipynb for a brief introduction of functionality
# or ./example.ipynb for an example analysis pipeline
```

## Example notebooks
- [Introduction with a simulated dataset](https://nbviewer.org/github/KangchengHou/calpred/blob/main/introduction.ipynb)
- [Example analysis pipeline](https://nbviewer.org/github/KangchengHou/calpred/blob/main/example.ipynb)