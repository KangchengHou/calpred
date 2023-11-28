# calpred.cli.R is for standalone use
# calpred.R is to be paired with calpred.fit.py

suppressPackageStartupMessages(library(statmod))


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