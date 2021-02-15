#!/usr/bin/env Rscript
suppressMessages(library(bnlearn))
suppressMessages(library(RcppCNPy))

args = commandArgs(trailingOnly=TRUE)
samples_filename = args[1]
amat_filename = args[2]
samples = npyLoad(samples_filename)
df = as.data.frame(samples)
colnames(df) = LETTERS[1:ncol(samples)]
a = npyLoad(amat_filename)
g = empty.graph(LETTERS[1:ncol(samples)])
amat(g) = a


s = score(g, df, type="bic-g")
npySave('tmp_output.npy', s)
