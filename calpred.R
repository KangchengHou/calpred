library(tibble)

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
compute_stats <- function(y, pred, group, predsd = NULL, predlow = NULL, predhigh = NULL, n_bootstrap = 50) {
  data <- data.frame(y, pred, group)

  if (!is.null(predsd)) {
    stopifnot(is.null(predlow) && is.null(predhigh))
    predlow <- pred - 1.645 * predsd
    predhigh <- pred + 1.645 * predsd
  }
  has_interval <- !is.null(predlow) && !is.null(predhigh)

  if (has_interval) {
    data <- data.frame(data, predlow, predhigh)
  }

  if (has_interval) {
    compute_metric <- function(data) {
      data %>%
        group_by(group) %>%
        summarize(
          r2 = cor(y, pred)**2,
          coverage = mean((y >= predlow) & (y <= predhigh)),
          length = mean((predhigh - predlow) / (2 * 1.645))
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

plot_stats <- function(stats, col = "r2") {
  if (is_tibble(stats)) {
    # Single stats
    stats_df <- stats %>%
      group_by(group) %>%
      summarize(
        mean = mean(.data[[col]]),
        sd = sd(.data[[col]])
      )

    p <- ggplot(stats_df, aes(x = as.factor(group), y = mean)) +
      geom_errorbar(aes(ymin = mean - sd * 1.96, ymax = mean + sd * 1.96),
        width = 0, position = position_dodge(width = 0.6)
      ) +
      geom_point() +
      xlab("Group") +
      ylab(col)
  } else {
    # Multiple stats, use name as group
    stats_df <- stats %>%
      bind_rows(.id = "name") %>%
      group_by(name, group, ) %>%
      summarize(
        mean = mean(.data[[col]]),
        sd = sd(.data[[col]]),
        .groups = "drop"
      )
    pd <- position_dodge(width = 0.3)
    p <- ggplot(stats_df, aes(x = as.factor(group), y = mean, group = name)) +
      geom_errorbar(aes(ymin = mean - sd * 1.96, ymax = mean + sd * 1.96, color = name),
        width = 0, position = pd
      ) +
      geom_point(aes(color = name), position = pd) +
      geom_line(aes(color = name), position = pd, linewidth = 0.2) +
      xlab("Group") +
      ylab(col) +
      theme(legend.position = c(0.7, 0.9), legend.background = element_rect(fill = "transparent")) +
      guides(color = guide_legend(title = NULL))
  }


  return(p)
}

normalize_table <- function(x) {
  return(data.frame(
    q = qnorm((rank(x) - 0.5) / length(x)),
    x = x
  ))
}
