source('R_algs/lrps.R')
source('R_algs/fit_lrps_path.R')

#!/usr/bin/env Rscript
suppressMessages(library(RcppCNPy))

args = commandArgs(trailingOnly=TRUE)
samples_filename = args[1]
samples_filename = "tmp_samples.npy"
samples = npyLoad(samples_filename)
Sigma = cor(samples)

path = fit.low.rank.plus.sparse.path(
  Sigma,
  gamma=.02,
  n=nrow(samples),
  max.iter = 100
)

for (i in 1:length(path)) {
  npySave(paste('tmp_out_S', i, '.npy', sep=""), path[[i]]$fit$S)
}

# npySave('tmp_out_L.npy', p$L)
# npySave('tmp_out_A.npy', p$A)
# npySave('tmp_out_U.npy', p$U)
