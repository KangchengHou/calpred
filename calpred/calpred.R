# calpred.cli.R is for standalone use
# calpred.R is to be paired with calpred.fit.py

suppressPackageStartupMessages(library(statmod))
suppressPackageStartupMessages(library(Rchoice))


#' @rdname train
#'
#' @title Train CalPred model
#'
#' @description A more detailed description of what the function is and how
#' it works. It may be a paragraph that should not be separated
#' by any spaces.
#'
#' @param mean_mat Covariates for mean effects \code{mean_mat}
#' @param sd_mat Covariates for variance effects \code{sd_mat}
#' @param y Response variable \code{y}
#'
#' @return Fitted parameters
#'
#' @export
#'
train_quant <- function(mean_mat, sd_mat, y, tol = 1e-6, maxit = 100) {
  fit <- statmod::remlscore(y = y, X = mean_mat, Z = sd_mat, tol = tol, maxit = maxit)
  mean_coef <- as.vector(fit$beta)
  sd_coef <- as.vector(fit$gamma)
  mean_se <- fit$se.beta
  sd_se <- fit$se.gam

  names(mean_coef) <- colnames(mean_mat)
  names(sd_coef) <- colnames(sd_mat)
  names(mean_se) <- colnames(mean_mat)
  names(sd_se) <- colnames(sd_mat)

  return(list(
    mean_coef = mean_coef,
    mean_se = mean_se,
    sd_coef = sd_coef,
    sd_se = sd_se
  ))
}

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
    sd_coef = sd_coef,
    sd_se = sd_se
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


model <- train_quant(mean_mat=x, sd_mat=z, y=y)

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
