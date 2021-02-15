source('R_algs/lrps.R')
source('R_algs/fit_lrps_path.R')

#' Perform K-fold cross-validation for the Low-Rank plus Sparse estimator
#' @import cvTools RSpectra Matrix
#' @export
cross.validate.low.rank.plus.sparse <- function(X,
                                                gamma,
                                                n,
                                                covariance.estimator=cor,
                                                n.folds = 5,
                                                lambdas = NULL,
                                                lambda.max=NULL,
                                                lambda.ratio=1e-4,
                                                n.lambdas=20,
                                                max.sparsity=0.5,
                                                max.rank=NA,
                                                tol=1e-05,
                                                max.iter=100,
                                                mu=0.1,
                                                verbose=FALSE,
                                                seed=NA) {

  Sigma <- covariance.estimator(X)
  p <- dim(Sigma)[1]

  if(is.null(lambdas)) {
    if (is.null(lambda.max)) {
      max.cor <- max(abs(Sigma - diag(diag(Sigma)))) * 2
      lambda.max <- max.cor / gamma
    }
    lambda.min <- lambda.max * lambda.ratio
    reason <- lambda.min / lambda.max
    lambdas <- lambda.max * reason **((0:n.lambdas)/n.lambdas)
  }

  # Start by computing the whole path on the full dataset.
  if (verbose) {
    print ("### Computing the path on the full dataset first ###")
  }
  path <- fit.low.rank.plus.sparse.path(Sigma, gamma, n, lambdas = lambdas,
                                        max.sparsity = max.sparsity,
                                        max.rank = max.rank,
                                        tol = tol, max.iter = max.iter, mu = mu,
                                        verbose = verbose)

  valid.lambdas <- c()
  for (i in 1:length(path)) {
    valid.lambdas <- c(valid.lambdas, path[[i]]$lambda)
  }
  if(!is.na(seed)) {
    set.seed(seed)
  }

  log.liks <- matrix(NA, ncol=2)
  for (lambda in valid.lambdas) {
    l1 <- gamma * lambda
    l2 <- (1 - gamma) * lambda
    fit <- NULL
    for (i in 1:length(path)) {
      if (path[[i]]$lambda == lambda) {
        fit <- path[[i]]$fit
        index <- i
        break()
      }
    }
    Strain <- covariance.estimator(X)
    fitll <- fit.low.rank.plus.sparse(Strain, l1, l2, nrow(X), init=fit, print_progress=F, tol=tol, maxiter=max.iter, mu=mu)
    if (fitll$termcode == -2) {
      ll <- NaN
    } else {
      # Compute the log likelihood on the testing set
      A <- fitll$S - fitll$L
      evals <- tryCatch({
      # In some cases this does not converge.
          RSpectra::eigs_sym(A, min(n-1,p-1))$values},
          error = function(e) {
              print(e)
              c(-10)
          })
      evals[abs(evals) < 1e-05] <- 1e-16
      if (any(evals < 0)) {
        ll <- NaN
      } else {
        gll <- sum(diag(Strain %*% A)) - sum(log(evals))
        ll <- path[[index]]$number.of.edges*log(n) + 4*path[[index]]$number.of.edges*gamma*log(ncol(X)) + 2*gll
      }
    }
    if (is.nan(ll)) {
      path[[index]] <- NULL
      next()
    }
    mean.ll <- ll
    sd.ll <- 0
    path[[index]]$mean_xval_ll <- mean.ll
    path[[index]]$sd_xval_ll <- sd.ll
    if(verbose) {
      print(paste("Lambda:", lambda, "X-Val Log-lik:",
                  mean.ll, "#Edges:",
                  path[[index]]$number.of.edges))
    }
  }

  path
}

#' @title Select a model based on the outcome of a K-fold cross-validation
#' @description  Selects the best model according to cross-validation. If the method is "min"
#' return the fit with the smallest average cross-validated log-likelihood.
#' If the method is "hastie", return the most parsimonious fit whose cross-validated
#' log-likelihood is within one standard deviation of the minimum.
#' @export
choose.cross.validate.low.rank.plus.sparse <- function(xval.path, method="min") {
  min.ll <- Inf
  min.sd <- NULL
  min.index <- NULL
  min.lambda <- NULL
  for (i in 1:length(xval.path)) {
    mean.ll <- xval.path[[i]]$mean_xval_ll
    if (mean.ll < min.ll) {
      min.ll <- mean.ll
      min.sd <- xval.path[[i]]$sd_xval_ll
      min.index <- i
      min.lambda <- xval.path[[i]]$lambda
    }
  }
  if (method == "min") {
    return(xval.path[[min.index]])
  }

  if (method == "hastie") {

    for (i in 1:length(xval.path)) {
      if (xval.path[[i]]$lambda < min.lambda) {
        next()
      }
      if (xval.path[[i]]$mean_xval_ll < (min.ll + min.sd)) {
        min.ll <- xval.path[[i]]$mean_xval_ll
        min.sd <- xval.path[[i]]$sd_xval_ll
        min.index <- i
        min.lambda <- xval.path[[i]]$lambda
      }
    }

    return(xval.path[[min.index]])

  } else {
    stop("Method has to be 'min' or 'hastie'.")
  }
}

#' Plot the output of a cross-validated low rank plus sparse model
#' @export
show.cross.validate.low.rank.plus.sparse <- function(lrps.path, ground.truth=NULL) {
  gamma <- lrps.path[[1]]$gamma
  lambdas <- ranks <- sparsities <- edges <- mean.lls <- sd.lls <- c()
  if (!is.null(ground.truth)) {
    ground.truth <- (ground.truth!=0) - diag(diag(ground.truth!=0))
  }
  prs <- rs <- c()
  for (i in 1:length(lrps.path)) {
    lambdas <- c(lambdas, lrps.path[[i]]$lambda)
    ranks <- c(ranks, lrps.path[[i]]$rank.L)
    edges <- c(edges, lrps.path[[i]]$number.of.edges)
    sparsities <- c(sparsities, lrps.path[[i]]$sparsity)
    mean.lls <- c(mean.lls, lrps.path[[i]]$mean_xval_ll)
    sd.lls <- c(sd.lls, lrps.path[[i]]$sd_xval_ll)

    if(!is.null(ground.truth)) {
      S <- lrps.path[[i]]$fit$S
      S <- (S!=0) - diag(diag(S!=0))
      if (sum(S) > 0) {
        pr <- sum(S * ground.truth) / sum(S)
        r <- sum(S * ground.truth) / sum(ground.truth)
        prs <- c(prs, pr)
        rs <- c(rs, r)
      }
    }
  }

  par(mfrow=c(2,2))
  plot(-log(lambdas), mean.lls, xlab="-Log10(Lambda)", ylab="Cross-Validated LogLik",
       ylim=c(min(mean.lls-sd.lls), max(mean.lls+sd.lls)))
  lines(-log(lambdas), mean.lls, col='red')
  lines(-log(lambdas), mean.lls+sd.lls, col='red', lty=3)
  lines(-log(lambdas), mean.lls-sd.lls, col='red', lty=3)
  plot(-log(lambdas), edges, xlab="-Log10(Lambda)", ylab="Number of Edges")
  plot(-log(lambdas), ranks, xlab="-Log10(Lambda)", ylab="Rank of L")
  if (!is.null(ground.truth)) {
    plot(rs, prs, xlab="Recall", ylab="Precision")
  }
  par(mfrow=c(1,1))
}


#!/usr/bin/env Rscript
suppressMessages(library(RcppCNPy))

args = commandArgs(trailingOnly=TRUE)
samples_filename = args[1]
samples = npyLoad(samples_filename)

path = cross.validate.low.rank.plus.sparse(
  samples,
  gamma=.02,
  n=nrow(samples),
  max.iter = 100
)
res = choose.cross.validate.low.rank.plus.sparse(path)
npySave('tmp_out_S.npy', res$fit$S)
# npySave('tmp_out_L.npy', p$L)
# npySave('tmp_out_A.npy', p$A)
# npySave('tmp_out_U.npy', p$U)
