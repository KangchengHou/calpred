#' @rdname simulate
#'
#' @title Simulate data using the model
#'
#' @description A more detailed description of what the function is and how
#' it works. It may be a paragraph that should not be separated
#' by any spaces.
#'
#' @param mean_mat Covariates for mean effects \code{mean_covar}
#' @param sd_mat Covariates for variance effects \code{var_covar}
#' @param mean_coef Coefficients for mean effects \code{mean_coef}
#' @param sd_coef Coefficients for variance effects \code{var_coef}
#'
#' @return Simulated data
#'
#' @export
#'
simulate <- function(mean_mat, sd_mat, mean_coef, sd_coef) {
  y_mean <- mean_mat %*% mean_coef
  y_sd <- sqrt(exp(sd_mat %*% sd_coef))
  y <- rnorm(n = length(y_mean), mean = y_mean, sd = y_sd)
  return(data.frame(y = y, mean = y_mean, sd = y_sd))
}

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

#' @rdname compute_stats
#'
#' @title Compute statistics
#'
#' @description A more detailed description of what the function is and how
#' it works. It may be a paragraph that should not be separated
#' by any spaces.
#'
#' @param y Response variable \code{y}
#' @param pred Covariates for mean effects
#' @param predsd Covariates for variance effects
#' @param group group information
#' @param n_bootstrap number of bootstrap
#'
#' @return Computed statistics
#'
#' @export
#'
compute_stats <- function(y, pred, group, predsd = NULL, n_bootstrap = 0) {
  data <- data.frame(y, pred, group)
  if (!is.null(predsd)) {
    stopifnot(nrow(data) == length(predsd))
    data$predsd <- predsd
  }
  if (!is.null(predsd)) {
    compute_metric <- function(data) {
      data %>%
        group_by(group) %>%
        summarize(
          r2 = cor(y, pred)**2,
          coverage = mean((y >= pred - 1.645 * predsd) & (y <= pred + 1.645 * predsd)),
          length = mean(predsd)
        )
    }
  } else {
    compute_metric <- function(data) {
      data %>%
        group_by(group) %>%
        summarize(
          r2 = cor(y, pred)**2
        )
    }
  }
  stats <- compute_metric(data)
  if (n_bootstrap <= 0) {
    return(stats)
  }
  bootstrap_stats <- list()
  for (i in 1:n_bootstrap) {
    bootstrap_stats[[i]] <- compute_metric(data[sample(nrow(data), replace = TRUE), ])
  }
  bootstrap_stats <- do.call(rbind, bootstrap_stats)
  return(list(stats = stats, bootstrap_stats = bootstrap_stats))
}
