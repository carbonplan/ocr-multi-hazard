################################################################################
# Per-cell standard errors of a trained FDP forest, run by fdp_train.py after
# train_fdp.R.
#
# The paper's standard errors (Collins et al. 2022, supplement 1.1.2): ranger's
# infinitesimal jackknife (Wager et al. 2014) with its empirical Bayes
# calibration, here computed for every cell of the watershed. Writes se.tif
# (float32) and se.json.
#
# The jackknife variance of a cell is p' G p / B^2 less ranger's bias
# correction, where p holds the cell's centred per-tree predictions and G is the
# B x B Gram matrix of the centred in-bag counts. G is formed once, so a cell
# costs one per-tree prediction and B^2 multiply-adds, independent of the
# training size; this matches ranger's rInfJack(calibrate = FALSE) to
# rounding.
#
# ranger fits its calibration prior to whatever batch it is handed, so its
# values shift with how cells are chunked. Here the prior and the Monte Carlo
# noise scale are fitted once per forest on a seeded random sample of the
# watershed's cells and then applied to every cell, which makes each cell's
# value independent of the chunking. The prior and noise scale are ranger's
# (gfit, and its half-forest estimate of sigma2).
#
# The calibrated variance is the exact posterior mean (ranger's gbayes) at
# each cell. ranger's calibrateEB means to interpolate it from the batch's 2%
# quantiles but passes `q =` to quantile(), which ignores it, so it
# interpolates linearly between the quartiles and the batch maximum; across
# the top quartile that overstates the standard error, by up to 0.05 in
# HUC 03.
#
# Usage:
#   Rscript se_fdp.R --model rf.rds --predictors DIR --prediction fdp.tif \
#     --out DIR [--threads 8] [--sample 100000] [--seed 20261030]
################################################################################

args <- commandArgs(trailingOnly = TRUE)
arg <- function(name, default = NULL) {
  i <- match(paste0("--", name), args)
  if (is.na(i)) {
    if (is.null(default)) stop("missing --", name)
    return(default)
  }
  args[[i + 1]]
}
out <- arg("out")
threads <- as.integer(arg("threads", "8"))
sample_size <- as.integer(arg("sample", "100000"))
seed <- as.integer(arg("seed", "20261030"))
piece <- 10000

suppressPackageStartupMessages({
  library(raster)
  library(ranger)
  library(parallel)
  library(jsonlite)
})

t0 <- Sys.time()
rf <- readRDS(arg("model"))
if (is.null(rf$inbag.counts)) stop("the forest was trained without keep.inbag")
vars <- c("floodplain", "distwater", "elevation", "hand", "nlcd", "pcurv", "hydric", "twi", "channel", "roads", "m3dp")
factor_cols <- c("nlcd", "floodplain")
predictors <- stack(file.path(arg("predictors"), paste0(vars, ".tif")))
names(predictors) <- vars
prediction <- raster(arg("prediction"))

N <- do.call(cbind, rf$inbag.counts)
B <- ncol(N)
n <- nrow(N)
if (max(N) == 1) stop("forests sampled without replacement need ranger's variance inflation, not implemented")
gram <- function(trees) crossprod(N[, trees, drop = FALSE] - rowMeans(N[, trees, drop = FALSE]))
n_var <- function(trees) {
  M <- N[, trees, drop = FALSE]
  mean(rowMeans(M^2) - rowMeans(M)^2)
}
G <- gram(seq_len(B))
G_var <- n_var(seq_len(B))

frame <- function(values) {
  d <- as.data.frame(values)
  for (v in factor_cols) d[[v]] <- factor(d[[v]], levels = rf$forest$covariate.levels[[v]])
  d
}

per_tree <- function(d) {
  p <- predict(rf, d, predict.all = TRUE, num.threads = 1, verbose = FALSE)$predictions
  p[, match("1", dimnames(p)[[2]]), ]
}

# Uncalibrated infinitesimal jackknife variance, as ranger:::rInfJack(calibrate = FALSE).
ij <- function(P, G, nv, nt) {
  Pc <- P - rowMeans(P)
  raw <- rowSums((Pc %*% G) * Pc) / nt^2
  raw - n * nv * rowSums(Pc^2) / nt / nt
}

# Forked workers on pieces of `piece` cells: ranger's per-tree prediction
# scales poorly across its own threads.
jackknife <- function(values, half = NULL) {
  if (!is.null(half)) {
    G_half <- gram(half)
    V_half <- n_var(half)
  }
  pieces <- mclapply(split(seq_len(nrow(values)), ceiling(seq_len(nrow(values)) / piece)), function(i) {
    P <- per_tree(frame(values[i, , drop = FALSE]))
    r <- cbind(mean = rowMeans(P), var = ij(P, G, G_var, B))
    if (!is.null(half)) r <- cbind(r, half = ij(P[, half, drop = FALSE], G_half, V_half, length(half)))
    r
  }, mc.cores = threads, mc.preschedule = FALSE)
  failed <- vapply(pieces, inherits, logical(1), "try-error")
  if (any(failed)) stop("jackknife worker failed: ", pieces[[which(failed)[1]]])
  do.call(rbind, pieces)
}

# Calibration fitted on a sample of the watershed's cells.
set.seed(seed)
all_rows <- seq_len(nrow(predictors))
sample_rows <- sort(sample(all_rows, min(length(all_rows), 400)))
pool <- do.call(rbind, lapply(sample_rows, function(r) getValues(predictors, r, 1)))
pool <- pool[complete.cases(pool), , drop = FALSE]
pool <- pool[sort(sample(nrow(pool), min(sample_size, nrow(pool)))), , drop = FALSE]
half <- sample(seq_len(B), ceiling(B / 2))
fit <- jackknife(pool, half)
var_full <- fit[, "var"]
var_half <- fit[, "half"]
delta <- length(half) / B
sigma2 <- (delta^2 + (1 - delta)^2) / (2 * (1 - delta)^2) * mean((var_half - var_full)^2)
prior <- ranger:::gfit(var_full, sqrt(sigma2))
posterior <- function(v) vapply(v, ranger:::gbayes, numeric(1), g.est = prior, sigma = sqrt(sigma2))
calib_x <- seq(min(prior$x), max(prior$x), length.out = 4000)
calib_y <- posterior(calib_x)
calibrate <- function(v) {
  y <- approx(calib_x, calib_y, xout = v)$y
  beyond <- is.na(y)
  if (any(beyond)) y[beyond] <- posterior(v[beyond])
  y
}
rm(fit, pool)
cat(sprintf(
  "calibration: %d cells, sigma2 %.3e, raw variance negative in %.1f%% (%.0fs)\n",
  length(var_full), sigma2, 100 * mean(var_full < 0), as.numeric(difftime(Sys.time(), t0, units = "secs"))
))

grid <- predictors[[1]]
se_out <- writeStart(
  raster(grid), file.path(out, "se.tif"), datatype = "FLT4S", overwrite = TRUE,
  options = c("COMPRESS=DEFLATE", "PREDICTOR=3", "TILED=YES", "BIGTIFF=YES")
)
rows <- max(1, floor(4e6 / ncol(grid)))
max_error <- 0
cells <- 0
negative <- 0
beyond <- 0
breaks <- c(seq(0, 0.5, by = 0.001), Inf)
counts <- numeric(length(breaks) - 1)
for (row in seq(1, nrow(grid), by = rows)) {
  m <- min(rows, nrow(grid) - row + 1)
  band <- getValues(predictors, row, m)
  keep <- which(complete.cases(band))
  result <- rep(NA_real_, nrow(band))
  if (length(keep)) {
    predicted <- getValues(prediction, row, m)[keep]
    jk <- jackknife(band[keep, , drop = FALSE])
    max_error <- max(max_error, abs(jk[, "mean"] - predicted), na.rm = TRUE)
    v <- jk[, "var"]
    negative <- negative + sum(v < 0)
    beyond <- beyond + sum(v < min(calib_x) | v > max(calib_x))
    result[keep] <- sqrt(pmax(calibrate(v), 0))
    counts <- counts + tabulate(findInterval(result[keep], breaks), length(counts))
    cells <- cells + length(keep)
  }
  se_out <- writeValues(se_out, result, row)
  cat(sprintf("[%s] rows %d-%d, %d cells so far\n", format(Sys.time(), "%H:%M:%S"), row, row + m - 1, cells))
}
se_out <- writeStop(se_out)
cat(sprintf("%d cells, max |mean of trees - fdp.tif| = %.2e\n", cells, max_error))
if (max_error > 1e-5) stop("per-tree predictions do not average to fdp.tif")

quantile_of <- function(q) breaks[which(cumsum(counts) >= q * cells)[1] + 1]
write_json(
  list(
    method = "infinitesimal jackknife (Wager et al. 2014), ranger rInfJack with empirical Bayes calibration",
    trees = B,
    training_rows = n,
    cells = cells,
    raw_negative_share = negative / cells,
    calibration = list(
      cells = length(var_full),
      seed = seed,
      sigma2 = sigma2,
      half_trees = length(half),
      cells_beyond_grid = beyond
    ),
    se_quantiles = list(
      p01 = quantile_of(0.01), p10 = quantile_of(0.1), p50 = quantile_of(0.5),
      p90 = quantile_of(0.9), p99 = quantile_of(0.99), max_bin = quantile_of(1)
    ),
    max_mean_error = max_error,
    seconds = as.numeric(difftime(Sys.time(), t0, units = "secs"))
  ),
  file.path(out, "se.json"), auto_unbox = TRUE, pretty = TRUE, digits = NA
)
cat("done\n")
