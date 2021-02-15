#!/usr/bin/env Rscript
suppressMessages(library(pcalg))
suppressMessages(library(RcppCNPy))

args = commandArgs(trailingOnly=TRUE)
samples_filename = args[1]
amat_filename = args[2]
samples = npyLoad(samples_filename)
amat = npyLoad(amat_filename)
g = as(amat, "GaussParDAG")

score = new("GaussL0penObsScore", samples, use.cpp=FALSE)
s = score$global.score(g)
npySave('tmp_output.npy', s)
