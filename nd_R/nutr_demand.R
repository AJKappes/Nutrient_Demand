# nutrient demand analysis

library(tidyverse)

remove(list = objects())
setwd('~/research/africa/Nutrient_Demand/nd_R/')

#### Data mgmt -----------------------------------------------------
df_bovine <- read.csv('df_bovine.csv')
df_goat <- read.csv('df_goat.csv')
df_sheep <- read.csv('df_sheep.csv')
df_list <- list(dfb = df_bovine, dfg = df_goat, dfs = df_sheep)

yvars <- c('protein_cons', 'fat_cons', 'carb_cons')
xvars <- c('income', 'protein_p_defl', 'fat_p_defl', 'carb_p_defl')
df_list <- lapply(df_list,
                  function(df)
                    mutate(df, income = OffFarmNetIncome + livincome))
Y_list <- lapply(df_list,
                 function(df)
                   select(df, yvars))
X_list <- lapply(df_list,
                 function(df)
                   select(df, xvars))

####  Gradient/Hessian functions ------------------------------------------------

# gradient function
Sfun <- function(theta, gamma, alpha, beta, Y, Xp, Xw) {
  
  k <- dim(Xp)[2]
  dtheta <- 0
  dgamma <- 0
  dalpha <- matrix(0, nrow = 1, ncol = k)
  dbeta <- matrix(0, nrow = k, ncol = k)
  
  # iterative summation across obs
  for (i in 1:length(Y)) {
    
    xpiter <- t(matrix(Xp[i, ]))
    Xpiter <- matrix(c(xpiter[1]*xpiter,
                       xpiter[2]*xpiter,
                       xpiter[3]*xpiter),
                     nrow = k, ncol = k)
    
    Zi <- c(Xw[i] - xpiter%*%alpha - 1/2*xpiter%*%beta%*%t(xpiter))
    resf <- c(Y[i] - (theta + xpiter%*%matrix(beta[1, ]) + gamma*Zi))
    
    dtheta <- dtheta - 2*resf
    dgamma <- dgamma - 2*Zi*resf
    dalpha <- dalpha + 2*gamma*xpiter*resf
    dbeta[1, 1] <- dbeta[1, 1] + 2*(gamma/2*Xpiter[1] - xpiter[1])*resf
    dbeta[1, 2:3] <- dbeta[1, 2:3] +
      2*(gamma/2*Xpiter[1, 2:3] - xpiter[2:3])*resf
    diag(dbeta)[2:3] <- diag(dbeta)[2:3] + gamma*Xpiter[c(5, 9)]*resf
    dbeta[2, 3] <- dbeta[2, 3] + gamma*Xpiter[2, 3]*resf
    
  }
  
  # set beta equality restrictions
  dbeta[2, 1] <- dbeta[1, 2]
  dbeta[3, 1] <- dbeta[1, 3]
  dbeta[3, 2] <- dbeta[2, 3]
  
  # Jacobian setup for Gauss-Newton NLS
  
  J <- matrix(0, nrow = length(Y), ncol = 14)
  R <- matrix(0, nrow = length(Y))
  for (i in 1:length(Y)) {

    xpiter <- t(matrix(Xp[i, ]))
    Xpiter <- matrix(c(xpiter[1]*xpiter,
                       xpiter[2]*xpiter,
                       xpiter[3]*xpiter),
                     nrow = k, ncol = k)

    Zi <- c(Xw[i] - xpiter%*%alpha - 1/2*xpiter%*%beta%*%t(xpiter))
    resf <- c(Y[i] - (theta + xpiter%*%matrix(beta[1, ]) + gamma*Zi))
    R[i] <- resf

    J[i, 1] <- -2*resf
    J[i, 2] <- -2*Zi*resf
    J[i, 3:5] <- 2*gamma*xpiter*resf
    J[i, 6] <- 2*(gamma/2*Xpiter[1] - xpiter[1])*resf
    J[i, 7:8] <- 2*(gamma/2*Xpiter[1, 2:3] - xpiter[2:3])*resf
    J[i, c(10, 14)] <- diag(dbeta)[2:3] + gamma*Xpiter[c(5, 9)]*resf
    J[i, 11] <- gamma*Xpiter[2, 3]*resf

  }
  
  J[, 9] <- J[, 7]
  J[, 12] <- J[, 8]
  J[, 13] <- J[, 11]
  
  Svec <- matrix(c(dtheta, dgamma, dalpha, dbeta))
  return(list(Svec, J , R))
  
}

# Hessian function
Hfun <- function(theta, gamma, alpha, beta, Y, Xp, Xw) {
  
  k <- dim(Xp)[2]
  
  # diagonal construction
  # del^2theta constant
  dtheta2 <- 2
  # iterative summation for the other H elements
  dgamma2 <- 0
  dalpha2 <- matrix(0, nrow = 1, ncol = k)
  dbeta2 <- matrix(0, nrow = k, ncol = k)
  lbeta <- length(dbeta2)
  
  # off-diagonal construction
  dthetad_ <- 0
  dgammad_ <- 0
  
  dalpha1d_ <- 0
  dalpha2d_ <- 0
  dalpha3d_ <- 0
  
  dbeta11d_ <- 0
  dbeta12d_ <- 0
  dbeta13d_ <- 0
  dbeta22d_ <- 0
  dbeta23d_ <- 0
  
  for (i in 1:length(Y)) {
    
    xpiter <- t(matrix(Xp[i, ]))
    Xpiter <- matrix(c(xpiter[1]*xpiter,
                       xpiter[2]*xpiter,
                       xpiter[3]*xpiter),
                     nrow = k, ncol = k)
    
    Zi <- c(Xw[i] - xpiter%*%alpha - 1/2*xpiter%*%beta%*%t(xpiter))
    resf <- c(Y[i] - (theta + xpiter%*%matrix(beta[1, ]) + gamma*Zi))
    
    B11p <- gamma/2*Xpiter[1] - xpiter[1]
    B1jp <- gamma/2*Xpiter[2:3] - xpiter[2:3]
    Bjkp <- gamma/2*Xpiter[4:lbeta]
    
    # diagonal
    dgamma2 <- dgamma2 + 2*Zi^2
    dalpha2 <- dalpha2 + 2*gamma^2*xpiter^2
    diag(dbeta2) <- diag(dbeta2) + 2*(gamma/2*diag(Xpiter) - xpiter)^2
    dbeta2[1, 2:3] <- dbeta2[1, 2:3] + 2*B1jp^2
    dbeta2[2, 3] <- dbeta2[2, 3] + gamma^2/2*Xpiter[2, 3]^2
    
    # off-diagonal
    dthetad_ <- dthetad_ +
      c(2*Zi, -2*gamma*xpiter, -2*B11p, -2*B1jp, -2*Bjkp)
    
    dgammad_ <- dgammad_ + 
      c(2*xpiter*resf - 2*gamma*xpiter*Zi,
        Xpiter[1]*resf - 2*B11p*Zi,
        Xpiter[2:3]*resf - 2*B1jp*Zi,
        Xpiter[4:lbeta]*resf - 2*Bjkp*Zi)
    
    dalpha1d_ <- dalpha1d_ +
      c(2*gamma^2*Xpiter[1, 2:3],
        2*gamma*xpiter[1]*B11p,
        2*gamma*xpiter[1]*B1jp,
        2*gamma*xpiter[1]*Bjkp)
    dalpha2d_ <- dalpha2d_ +
      c(2*gamma^2*Xpiter[2, 3],
        2*gamma*xpiter[2]*B11p,
        2*gamma*xpiter[2]*B1jp,
        2*gamma*xpiter[2]*Bjkp)
    dalpha3d_ <- dalpha3d_ +
      c(2*gamma*xpiter[3]*B11p,
        2*gamma*xpiter[3]*B1jp,
        2*gamma*xpiter[3]*Bjkp)
    
    dbeta11d_ <- dbeta11d_ +
      c(2*B11p*B1jp, 2*B11p*Bjkp)
    dbeta12d_ <- dbeta12d_ +
      c(2*B1jp[1]*B1jp[2], 2*B1jp[1]*Bjkp)
    dbeta13d_ <- dbeta13d_ +
      c(2*B1jp[2]*Bjkp)
    dbeta22d_ <- dbeta22d_ +
      c(gamma*Xpiter[2, 2]*Bjkp[3:length(Bjkp)])
    dbeta23d_ <- dbeta23d_ +
      c(2*Bjkp[3]*Bjkp[4:length(Bjkp)])
    
  }
  
  # assign symmetric Hessian beta elements
  dbeta2[2, 1] <- dbeta2[1, 2]
  dbeta2[3, 1] <- dbeta2[1, 3]
  dbeta2[3, 2] <- dbeta2[2, 3]
  
  dbeta21d_ <- dbeta12d_[3:length(dbeta12d_)]
  dbeta31d_ <- dbeta13d_[5:length(dbeta13d_)]
  dbeta32d_ <- dbeta23d_[3:length(dbeta23d_)]
  
  # construct diagonal and off-diagonal matrix elements
  diagonal <- c(dtheta2, dgamma2, dalpha2, dbeta2)
  d2list <- list(dthetad_, dgammad_,
                 dalpha1d_, dalpha2d_, dalpha3d_,
                 dbeta11d_, dbeta12d_, dbeta13d_,
                 dbeta21d_, dbeta22d_, dbeta23d_,
                 dbeta31d_, dbeta32d_)
  #print(d2list)
  
  Hmat <- diag(diagonal)
  
  j <- dim(Hmat)[1]

  # using upper.tri() does not fill by rows
  # having to use for loop to fill upper tri row vecs
  for (i in 1:(j - 1)) {

    Hmat[i, (i + 1):j] <- d2list[[i]]

  }

  # set symmetric lower off-diagonal
  Hmat[lower.tri(Hmat)] <- t(Hmat)[lower.tri(Hmat)]
  return(Hmat)
  
}

#### Data ----------------------------------------------------------

y <- Y_list[['dfb']]$protein_cons
xp <- X_list[['dfb']] %>%
  select(contains('_p_defl')) %>% 
  as.matrix() %>% 
  unname()
xw <- X_list[['dfb']]$income

param_vals <- list(theta = 1, gamma = 1,
                   alpha = matrix(c(1, 1, 1)),
                   beta = matrix(c(1, 1, 1,
                                   1, 1, 1,
                                   1, 1, 1),
                                 ncol = 3, byrow = TRUE))

data_vals <- list(Y = y, Xp = xp, Xw = xw)

#### Newton-Rhapson optimization -----------------------------------

err <- 1
tol <- .001
itr <- 1
max_itr <- 100
param_mat <- matrix(0, nrow = max_itr, ncol = 14)
while(err > tol & itr <= max_itr) {
  
  S <- do.call(Sfun, c(param_vals, data_vals))[[1]]
  H <- do.call(Hfun, c(param_vals, data_vals))
  
  params_m <- matrix(unlist(param_vals))
  params_mp1 <- params_m - solve(H)%*%S
  
  param_mat[itr, ] <- c(params_mp1)
  param_vals <- list(theta = params_mp1[1], gamma = params_mp1[2],
                     alpha = matrix(params_mp1[3:5]),
                     beta = matrix(params_mp1[6:14], ncol = 3, byrow = TRUE))
  
  err <- max(abs(params_mp1 - params_m))
  itr <- itr + 1
  
  cat('Iteration', itr - 1,
      '\nMax optimization error', err,
      '\n')
  
}

#### Testing -------------------------------------------------------

J <- do.call(Sfun, c(param_vals, data_vals))[[2]]
R <- do.call(Sfun, c(param_vals, data_vals))[[3]]
params_m <- matrix(unlist(param_vals))
params_mp1 <- params_m - solve(t(J)%*%J)%*%t(J)%*%R


S <- matrix(do.call(Sfun, c(param_vals, data_vals))[[1]][-c(3:5)])
H <- do.call(Hfun, c(param_vals, data_vals))[-c(3:5) ,-c(3:5)]
params_m <- matrix(unlist(param_vals))
params_mp1 <- params_m - solve(H)%*%S
param_vals <- list(theta = params_mp1[1], gamma = params_mp1[2],
                   alpha = matrix(params_mp1[3:5]),
                   beta = matrix(params_mp1[6:14], ncol = 3, byrow = TRUE))

data <- data.frame(y = y, xp = xp, xw = xw)
summary(lm(y~., data = data))
