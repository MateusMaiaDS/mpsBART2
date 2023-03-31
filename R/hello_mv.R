rm(list=ls())
library(tidyverse)

Rcpp::sourceCpp("src/mpsBART.cpp")
source("R/other_functions.R")
source("R/wrap_bart.R")
source("R/bayesian_simulation.R")
n_ <- 250
set.seed(42)

# Simulation 1
fried_sim <- mlbench::mlbench.friedman1(n = n_,sd = 0.01)
friedman_no_interaction <- function (n, sd = 1)
{
     x <- matrix(runif(4 * n), ncol = 4)
     y <- 10 * sin(pi * x[, 1] )
     y <- y + 20 * (x[, 2] - 0.5)^2 + 10 * x[, 3] + 5 * x[, 4]
     if (sd > 0) {
          y <- y + rnorm(n, sd = sd)
     }
     list(x = x, y = y)
}

sd_ <- 5
fried_sim <- friedman_no_interaction(n = n_,sd = sd_)
fried_sim_new_sample <- friedman_no_interaction(n = n_,sd = sd_)

x <- fried_sim$x[,,drop = FALSE]
x_new <- fried_sim_new_sample$x
y <- fried_sim$y

# Transforming into data.frame
x <- as.data.frame(x)
x_test <- as.data.frame(x_new)


# Testing the mpsBART
bart_test <- rbart(x_train = x,y = unlist(c(y)),x_test = x_test,
                   n_tree = 1,n_mcmc = 2500,alpha = 0.95,dif_order = 0,
                   beta = 2,nIknots = 10,delta = 1,
                   n_burn = 500,scale_bool = FALSE)


# Running BART
bartmod <- dbarts::bart(x.train = x,y.train = unlist(c(y)),ntree = 200,x.test = x_test,keeptrees = TRUE)

# Convergence plots
par(mfrow = c(1,2))
plot(bart_test$tau_post,type = "l", main = expression(tau),ylab=  "")
plot(bartmod$sigma^-2, type = "l", main = paste0("BART: ",expression(tau)),ylab=  "")

par(mfrow = c(1,2))
plot(bart_test$y_hat %>% rowMeans(),y, main = 'mpsBART', xlab = "mpsBART pred", ylab = "y")
plot(bartmod$yhat.train.mean,y, main = "BART", xlab = "BART pred", ylab = "y")

# Comparing on the test set
pred_bart <- colMeans(predict(bartmod,fried_sim_new_sample$x))

# Storing the results
rmse(x = fried_sim_new_sample$y,y = bartmod$yhat.test.mean)
rmse(x = fried_sim_new_sample$y,y = rowMeans(bart_test$y_hat_test))

par(mfrow=c(1,1))
plot(bartmod$yhat.test.mean,rowMeans(bart_test$y_hat_test))

