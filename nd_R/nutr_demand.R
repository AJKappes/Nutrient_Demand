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

#### IDS estimation ------------------------------------------------

# Newton-Rhapson setup

# gradient function
Sfun <- function(theta, gamma, phi, alpha, beta, Y, Xp, Xw) {
  
  k <- dim(Xp)[2]
  dtheta <- 0
  dgamma <- 0
  dphi <- matrix(0, nrow = 1, ncol = k)
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
    resf <- c(Y[i] - (theta + xpiter%*%phi + gamma*Zi))
    
    dtheta <- dtheta + -2*resf
    dgamma <- dgamma + -2*Zi*resf
    dphi <- dphi + -2*xpiter*resf
    dalpha <- dalpha + 2*gamma*xpiter*resf
    dbeta <- dbeta + gamma*Xpiter*resf
    
  }
  
  Svec <- matrix(c(dtheta, dgamma, dphi, dalpha, dbeta))
  return(Svec)
  
}

# Hessian function
Hfun <- function(theta, gamma, phi, alpha, beta, Y, Xp, Xw) {
  
  k <- dim(Xp)[2]
  
  # diagonal construction
  # del^2theta constant
  dtheta2 <- 2
  # iterative summation for the other H elements
  dgamma2 <- 0
  dphi2 <- matrix(0, nrow = 1, ncol = k)
  dalpha2 <- matrix(0, nrow = 1, ncol = k)
  dbeta2 <- matrix(0, nrow = k, ncol = k)
  lbeta <- length(dbeta2)
  
  # off-diagonal construction
  dthetad_ <- 0
  dgammad_ <- 0
  
  dphi1d_ <- 0
  dphi2d_ <- 0
  dphi3d_ <- 0
  
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
    Xpiterd_ <- matrix(c(xpiter[1]*xpiter,
                         xpiter[2]*xpiter,
                         xpiter[3]*xpiter),
                       nrow = k, ncol = k)
    
    Zi <- c(Xw[i] - xpiter%*%alpha - 1/2*xpiter%*%beta%*%t(xpiter))
    resf <- c(Y[i] - (theta + xpiter%*%phi + gamma*Zi))
    
    # diagonal
    dgamma2 <- dgamma2 + 2*Zi^2
    dphi2 <- dphi2 + 2*xpiter^2
    dalpha2 <- dalpha2 + 2*gamma^2*xpiter^2
    dbeta2 <- dbeta2 + 1/2*gamma^2*Xpiterd_^2
    
    # off-diagonal
    dthetad_ <- dthetad_ +
      c(2*Zi, 2*xpiter, -2*gamma*xpiter, -gamma*Xpiterd_)
    
    dgammad_ <- dgammad_ + 
      c(2*xpiter*Zi, 2*xpiter*resf - 2*gamma*xpiter*Zi,
        resf*Xpiterd_ - gamma*Zi*Xpiterd_)
    
    dphi1d_ <- dphi1d_ +
      c(2*Xpiterd_[1, 2:3], -2*gamma*Xpiterd_[1, ], -gamma*xpiter[1]*Xpiterd_)
    dphi2d_ <- dphi2d_ + 
      c(2*Xpiterd_[2, 3], -2*gamma*Xpiterd_[2, ], -gamma*xpiter[2]*Xpiterd_)
    dphi3d_ <- dphi3d_ +
      c(-2*gamma*Xpiterd_[3, ], -gamma*xpiter[3]*Xpiterd_)
    
    dalpha1d_ <- dalpha1d_ +
      c(2*gamma^2*Xpiterd_[1, 2:3], gamma^2*xpiter[1]*Xpiterd_)
    dalpha2d_ <- dalpha2d_ +
      c(2*gamma^2*Xpiterd_[2, 3], gamma^2*xpiter[2]*Xpiterd_)
    dalpha3d_ <- dalpha3d_ +
      c(gamma^2*xpiter[3]*Xpiterd_)
    
    # problems with proper number of elements
    dbeta11d_ <- dbeta11d_ +
      c(1/2*gamma^2*Xpiterd_[1, 1]*Xpiterd_[2:lbeta])
    dbeta12d_ <- dbeta12d_ +
      c(1/2*gamma^2*Xpiterd_[1, 2]*Xpiterd_[3:lbeta])
    dbeta13d_ <- dbeta13d_ +
      c(1/2*gamma^2*Xpiterd_[1, 3]*Xpiterd_[4:lbeta])
    dbeta22d_ <- dbeta22d_ +
      c(1/2*gamma^2*Xpiterd_[2, 2]*Xpiterd_[6:lbeta])
    dbeta23d_ <- dbeta23d_ +
      c(1/2*gamma^2*Xpiterd_[2, 3]*Xpiterd_[7:lbeta])
    
  }
  
  # assign symmetric Hessian beta elements
  dbeta21d_ <- dbeta12d_[3:length(dbeta12d_)]
  dbeta31d_ <- dbeta13d_[5:length(dbeta13d_)]
  dbeta32d_ <- dbeta23d_[3:length(dbeta23d_)]
  
  # construct diagonal and off-diagonal matrix elements
  diagonal <- c(dtheta2, dgamma2, dphi2, dalpha2, dbeta2)
  d2list <- list(dthetad_, dgammad_,
                 dphi1d_, dphi2d_, dphi3d_,
                 dalpha1d_, dalpha2d_, dalpha3d_,
                 dbeta11d_, dbeta12d_, dbeta13d_,
                 dbeta21d_, dbeta22d_, dbeta23d_,
                 dbeta31d_, dbeta32d_)
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

# initial test of nonlinear estimation

y <- Y_list[['dfb']]$protein_cons
xp <- X_list[['dfb']] %>%
  select(contains('_p_defl')) %>% 
  as.matrix() %>% 
  unname()
xw <- X_list[['dfb']]$income

y <- y[1:10]
xw <- xw[1:10]
xp <- xp[1:10, ]


param_vals <- list(theta = 1, gamma = 1,
                   phi = matrix(c(-1, 1, 1), nrow = 3, ncol = 1),
                   alpha = matrix(c(-1, 1, 1), nrow = 3, ncol = 1),
                   beta = diag(rep(-1, 3)))

data_vals <- list(Y = y, Xp = xp, Xw = xw)

S <- do.call(Sfun, c(param_vals, data_vals))
H <- do.call(Hfun, c(param_vals, data_vals))

solve(H)
chol(H)
qr.solve(H)
s <- svd(H)
s$v%*%solve(diag(s$d))%*%t(s$u)
