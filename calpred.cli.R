#!/usr/bin/env Rscript

# Load required libraries
library(optparse)
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
train <- function(mean_mat, sd_mat, y, tol = 1e-6, maxit = 100) {
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


#' @rdname predict
#'
#' @title Predict with CalPred model
#'
#' @description A more detailed description of what the function is and how
#' it works. It may be a paragraph that should not be separated
#' by any spaces.
#'
#' @param mean_mat Covariates for mean effects \code{mean_mat}
#' @param sd_mat Covariates for variance effects \code{sd_mat}
#' @param y Response variable \code{y}
#'
#' @return Predicted mean and SD
#'
#' @export
#'
predict <- function(mean_mat, sd_mat, mean_coef, sd_coef) {
  if (!all.equal(colnames(mean_mat), names(mean_coef))) {
    stop("colnames(mean_mat) != names(mean_coef)")
  }
  if (!all.equal(colnames(sd_mat), names(sd_coef))) {
    stop("colnames(sd_mat) != names(sd_coef)")
  }
  y_mean <- mean_mat %*% mean_coef
  y_var <- exp(sd_mat %*% sd_coef)
  return(data.frame(
    mean = y_mean,
    sd = sqrt(y_var)
  ))
}

predict_probit <- function(mean_mat, sd_mat, mean_coef, sd_coef) {
  if (!all.equal(colnames(mean_mat), names(mean_coef))) {
    stop("colnames(mean_mat) != names(mean_coef)")
  }
  if (!all.equal(colnames(sd_mat), names(sd_coef))) {
    stop("colnames(sd_mat) != names(sd_coef)")
  }
  y_mean <- mean_mat %*% mean_coef
  y_var <- exp(sd_mat %*% sd_coef)
  y_prob <- pnorm(y_mean / y_var)
  return(data.frame(
    prob = y_prob,
    mean = y_mean
  ))
}

option_list <- list(
  make_option(
    c("--df"),
    type = "character",
    help = "Path to the data file (tsv format)"
  ),
  make_option(
    c("--y_col"),
    type = "character",
    help = "Name of the column containing the target variable",
  ),
  make_option(
    c("--out_prefix"),
    type = "character",
    help = "Output file path for the model"
  ),
  make_option(
    c("--mean_cols"),
    type = "character",
    default = "",
    help = "Comma-separated list of column names for mean predictors"
  ),
  make_option(
    c("--sd_cols"),
    type = "character",
    default = "",
    help = "Comma-separated list of column names for standard deviation predictors"
  )
)

# Parse command-line arguments
opt <- parse_args(OptionParser(option_list = option_list))

data <- read.table(opt$df, header = TRUE, sep = "\t")
y <- data[, opt$y_col]


# Extract mean_mat and sd_mat
mean_cols <- strsplit(opt$mean_cols, ",")[[1]]
sd_cols <- strsplit(opt$sd_cols, ",")[[1]]

mean_mat <- as.matrix(
  cbind(
    const = 1, data[, c(which(names(data) %in% mean_cols)), drop = FALSE]
  )
)

binary_trait_flag <- all(y %in% c(0, 1, TRUE, FALSE))

if (binary_trait_flag) {
  print(
    paste0(opt$y_col, " column contains binary trait (0/1). Using probit regression.")
  )
  sd_mat <- as.matrix(
    data[, c(which(names(data) %in% sd_cols)), drop = FALSE]
  )
  train_func <- train_probit
} else {
  print(
    paste0(opt$y_col, " column contains values other than 0/1. Using continuous regression.")
  )
  sd_mat <- as.matrix(
    cbind(
      const = 1, data[, c(which(names(data) %in% sd_cols)), drop = FALSE]
    )
  )
  train_func <- train
}


# model training
model <- train_func(mean_mat = mean_mat, sd_mat = sd_mat, y = y)

# save coefficients
rows <- unique(unlist(sapply(model, names)))
model_df <- data.frame(sapply(model, function(coefs) {
  coefs[match(rows, names(coefs))]
}), row.names = NULL)
model_df <- cbind(name = rows, model_df)

write.table(
  model_df,
  file = paste0(opt$out_prefix, ".coef.tsv"),
  row.names = FALSE, col.names = TRUE, sep = "\t", na = "NA"
)
print(paste0("Coefficients saved to ", opt$out_prefix, ".coef.tsv"))

# fit back to the data
fitted_df <- cbind(
  data[, 1, drop = FALSE], # first column as index
  predict(
    mean_mat = mean_mat, sd_mat = sd_mat,
    mean_coef = model$mean_coef, sd_coef = model$sd_coef
  )
)
write.table(
  fitted_df,
  file = paste0(opt$out_prefix, ".fitted.tsv"),
  row.names = FALSE, col.names = TRUE, sep = "\t", na = "NA"
)
print(paste0("Fitted values saved to ", opt$out_prefix, ".fitted.tsv"))
