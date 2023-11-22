# calpred.cli.R is for standalone use
# calpred.R is to be paired with calpred.fit.py

suppressPackageStartupMessages(library(statmod))
suppressPackageStartupMessages(library(Rchoice))


train_probit <- function(mean_mat, sd_mat, y) {
  # mean_mat is assumed to be already containing intercept
  # sd_mat cannot contain intercept, otherwise the model will be unidentified
  # because a constant can be multiplied to both x^T b and exp(z^T gamma)
  model <- hetprob(y ~ mean_mat - 1 | sd_mat, link = "probit")
  mf <- model$mf
  X <- model.matrix(model$formula, data = mf, rhs = 1)
  Z <- model.matrix(model$formula, data = mf, rhs = 2)
  K <- ncol(X)

  mean_coef <- coef(model)[1:K]
  sd_coef <- coef(model)[-c(1:K)]

  se <- sqrt(diag(vcov(model)))
  mean_se <- se[1:K]
  sd_se <- se[-c(1:K)]
  names(mean_coef) <- colnames(mean_mat)
  names(mean_se) <- colnames(mean_mat)
  names(sd_coef) <- colnames(sd_mat)
  names(sd_se) <- colnames(sd_mat)

  print(summary(model))
  return(list(
    mean_coef = mean_coef,
    mean_se = mean_se,
    sd_coef = sd_coef * 2,
    sd_se = sd_se * 2
  ))
}

# Command line arguments
y_file <- commandArgs(TRUE)[1]
x_file <- commandArgs(TRUE)[2]
z_file <- commandArgs(TRUE)[3]
out_prefix <- commandArgs(TRUE)[4]

# Load vector y, matrices x, and z from files
y <- as.numeric(scan(y_file, quiet=TRUE))
x <- as.matrix(read.table(x_file, sep=" "))
z <- as.matrix(read.table(z_file, sep=" "))

model <- train_probit(mean_mat=x, sd_mat=z, y=y)

# save coefficients
write.table(
    cbind(model$mean_coef, model$mean_se),
    file=paste0(out_prefix, ".mean"), 
    row.names=FALSE, col.names=FALSE
)
write.table(
    cbind(model$sd_coef, model$sd_se),
    file=paste0(out_prefix, ".sd"), 
    row.names=FALSE, col.names=FALSE
)