vec.required_packages <- c("foreach", "doParallel")
vec.new_packages <- vec.required_packages[!(vec.required_packages %in% installed.packages()[,"Package"])]
if(length(vec.new_packages)) install.packages(vec.new_packages)
library(foreach)
library(doParallel)


fn.amnfis <- function(X, d, clusters){

  k <- nrow(clusters)

  ################ HELPER FUNCTIONS ###################

  fn.object_to_params <- function(object){
    return(c(c(object$phi_0), c(object$PHI)))
  }

  fn.params_to_object <- function(object, params, k, n, m){
    i <- 1

    object$phi_0 <- c(params[i: (i + k -1)])
    i <- i + k

    object$PHI <- matrix(data = params[i:(i + k * n - 1)],
                         nrow = k, ncol = n)
    i <- i + k * n

    return(object)
  }

  fn.optim <- function(v){
    obj <- fn.params_to_object(obj, v, k, n, m)
    y <- fn.amnfis_simulate(obj, X, clusters)
    y[y == 0] = 0.00001
    y[y == 1] = 0.99999
    # error = sum((d - y)^2) # MSE
    error <- -sum(d * log(y) + (1 - d) * log(1 - y)) # Cross entropy
    return(error)
  }

  ################### MAIN FUNCTION ###############
  m <- dim(X)[1]
  n <- dim(X)[2]
  obj <- NULL

  ########INITIALIZE NUERAL NETWORKS PARAMS (WEIGHTS) TO RANDOM VALUES (PHI_O Y PHI)###########
  obj$phi_0 <- fn.load_random_vector(k)
  obj$PHI <- fn.load_random_phi(k,n)

  ################## BEGIN TRAINING #####################

  v <- fn.object_to_params(obj)

  convergence <- FALSE
  while(convergence == FALSE){
    optimizer <- optim(v, fn.optim, method = 'BFGS')
    if(optimizer$convergence == 0){
      convergence <- TRUE
    }else{
      v <- optimizer$par
    }
  }

  obj <- fn.params_to_object(obj, optimizer$par, k, n, m)
  return(obj)

}

fn.amnfis_simulate <- function(obj, X, C) {

  n <- dim(X)[2]
  # Layer 1: calculate the distance between the Inputs(Xi) and each Cluster(Ci)
  DISTANCES <- fn.get_Xi_Ci_Distances(X, C)
  # Layer 2: calculate the membership function from each Input to each Cluster
  MEMBERSHIP = fn.get_membership(DISTANCES)
  # Layer 3: calculate the contribution of each Cluster
  CONTRIBUTIONS <- fn.contrib(MEMBERSHIP)
  # Layer 4: calculate the output for each cluster
  X_PHI = X %*% t(obj$PHI)
  phi_0_matrix <- matrix(rep(obj$phi_0, times = nrow(X_PHI)), ncol = length(obj$phi_0), byrow = TRUE)
  PHI_0_X_PHI <- phi_0_matrix + X_PHI
  # Layer 5: calculate the output as the weighted average from the two previous layers
  X_PHI_CONTRIBUTIONS <- CONTRIBUTIONS * PHI_0_X_PHI
  y <- rowSums(X_PHI_CONTRIBUTIONS)

  # y[y<-5] <- 0.01
  # y[y>5] <-  0.99
  # y[y>= -5 && y<=5] <- 1/(1+exp(-y))
  y <- 1/(1+exp(-y))

  return(y)
}

fn.load_random_vector <- function(size){
  return(rnorm(size))
}

fn.load_random_phi <- function(k ,n){
  phi_params <- matrix(rnorm(k * n), nrow = k, ncol = n)
  return(phi_params)
}

fn.get_Xi_Ci_Distances <- function(X, C){
  m <- dim(X)[1]
  n <- dim(X)[2]
  k <- dim(C)[1]
  replByCol <- rep(k, n)
  replByRow <- rep(m, n)
  transformedX <- X[,rep(1:n, replByCol)]
  transformedC <- matrix(rep(C, each=nrow(X)), nrow=m)
  dista <- (transformedX - transformedC)^2
  distaces3D <- array(dista, c(m,k,n))
  distancesList <- lapply(seq(dim(distaces3D)[3]), function(x) distaces3D[ , , x])
  rsp <- Reduce('+', distancesList)
  return(rsp)
}

fn.get_membership <- function(distances){
  if(is.vector(distances)){
    distances <- matrix(distances)
  }
  return(exp(-distances/rowSums(distances)))
}

fn.contrib <- function(membership){
  return(membership/rowSums(membership))
}


fn.transform_output <- function(output){
  rsp <- ifelse(as.vector(output) > 0.5, 1, 0)
  return(rsp)
}

# Function to define the centroids of each cluster
fn.getcentroids <- function(all_data, n, formula){
  print("Calculating centroids for clusters...")

  fn.groupdata <- function(all_data, n, formula){

    if(n ==1){
      data <- list(all_data)
    }else{
      models <- list()

      initial_model <- fn.splitdata(formula, all_data)
      initial_model <- initial_model[[1]]
      part1 <- initial_model$data[initial_model$part1,]
      part2 <- initial_model$data[initial_model$part2,]

      idx_loop <- 3

      while(idx_loop <= n){
        model_a <- fn.splitdata(formula, part1)
        model_b <- fn.splitdata(formula, part2)
        model_a <- model_a[[1]]
        model_b <- model_b[[1]]

        if(length(models) == 0){
          models <- list(model_a, model_b)
        }else{
          models[[length(models) + 1]] <- model_a
          models[[length(models) + 1]] <- model_b
        }
        errors <- lapply(models, fn.fetcherrors)
        idx_min_error <- which.min(errors)
        obj <- models[[idx_min_error]]
        models[[idx_min_error]] <- NULL
        part1 <- obj$data[obj$part1,]
        part2 <- obj$data[obj$part2,]
        idx_loop <- idx_loop + 1
      }
      data <- lapply(models, fn.datafrommodels)
      data[[length(data) + 1 ]] <- part1
      data[[length(data) + 1 ]] <- part2
    }
    return(data)
  }

  fn.splitdata <- function(formula, data){
    original <- data
    mat <- apply(as.matrix(data[,1:ncol(data) -1]), 2, fn.getpartitions, formula = formula, data = data)
    errors <- unlist(lapply(mat, fn.fetcherrors))
    lowest_error_idx <- which.min(errors)
    return(unname(mat[lowest_error_idx]))
  }

  fn.fetcherrors <- function(object){
    return(object$error)
  }

  fn.datafrommodels <- function(model){
    return(model$data)
  }

  fn.getpartitions <- function(v, formula, data){
    v_replicated <- fn.replicatedata(v)
    partitions <- t(ifelse(t(v_replicated) >= v, 1, 0))
    fit_errors <- apply(partitions, 2, fn.fitdata, formula = formula, data = data)
    errors <- lapply(fit_errors, fn.fetcherrors)
    errors <- unlist(errors)
    lowes_error_idx <- which.min(errors)
    best_partition <- fit_errors[[lowes_error_idx]]
    return(best_partition)
  }

  # Split into two datasets, for training and for getting the errors. this is done for each column
  fn.fitdata <- function(v_logic, formula, data){
    part1 <- which(v_logic == 1, arr.ind = TRUE)
    part2 <- which(v_logic == 0, arr.ind = FALSE)
    data_part1 <- data[part1,]
    data_part2 <- data[part2,]
    error1 <- ifelse((nrow(data_part1) > 0 && nrow(data_part2) > 0), fn.fit_linear_classifier(formula, data_part1), 100)
    error2 <- ifelse((nrow(data_part1) > 0 && nrow(data_part2) > 0), fn.fit_linear_classifier(formula, data_part2), 100)
    obj <- list(error = error1 + error2, part1 = part1, part2 = part2, data = data)
    return(obj)
  }

  # Replicates data
  fn.replicatedata <- function(v){
    n <- length(v)
    return(matrix(rep(v, each = n), ncol = n, byrow = TRUE))
  }

  # Removes the class columns
  fn.removeclass <- function(v){
    v <- v[1:length(v) - 1]
    return(v)
  }

  # Fits a linear classifier for a sample data and returns the error
  fn.fit_linear_classifier <- function(formula, data){
    y <- data[,ncol(data)]
    response <- glm(formula, data = data)
    mse <- fn.error(y, response$fitted.values)
    return(mse)
  }

  # Calculates the MSE
  fn.error <- function(y, predictions){
    mse <- sum((y - predictions)^2)/length(y)
    return(mse)
  }

  groups <- fn.groupdata(all_data, n, formula)
  centroids <- lapply(groups, colMeans)
  columns <- ncol(all_data) - 1
  centroids <- lapply(centroids, fn.removeclass)
  centroid_matrix <- matrix(unlist(centroids), ncol = columns, byrow = TRUE)
  print("Centroids calculated...")
  return(centroid_matrix)
}

fn.train_amnfis <- function(df, X, d, formula){
  centroids <- fn.getcentroids(all_data = df, n = 2, formula = formula)
  model <- fn.amnfis(X = X, d = d, clusters = centroids)
  predict <- fn.amnfis_simulate(obj = model, X = X, C = centroids)
  predict <- fn.transform_output(predict)
  acc <- length(d[d == predict]) / length(d)
  result <- NULL
  result$acc <- c(acc)
  result$model <- c(model)
  for (j in 3:10) {
    print(paste("iteration ", j))
    acc <- 0
    vec_best_data_point <- c()
    registerDoParallel(4)
    response <- foreach (i = 1:nrow(X)) %dopar% {
      vec_data_point <- X[i,]
      centroids_tmp <- rbind(centroids, vec_data_point)
      model <- fn.amnfis(X = X, d = d, clusters = centroids_tmp)
      predict <- fn.amnfis_simulate(obj = model, X = X, C = centroids_tmp)
      predict <- fn.transform_output(predict)
      accuracy <- length(d[d == predict]) / length(d)
      my_res <- NULL
      my_res$acc <- accuracy
      my_res$centroid <- vec_data_point
      my_res$model <- model
      my_res
    }
    stopImplicitCluster()
    best_record <- response[[which.max(lapply(response, function(x) x$acc))]]
    vec_best_data_point <- best_record$centroid
    acc <- best_record$acc
    mdl <- best_record$model

    centroids <- rbind(centroids, vec_best_data_point)
    result$acc <- c(result$acc, acc)
    result$model <- c(result$model, mdl)
    print(paste("centroids ", nrow(centroids)))
  }
  result$clusters <- centroids
  return(result)
}
