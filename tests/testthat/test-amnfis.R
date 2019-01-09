context("amnfis layers, forward propagation")
library(amnfisClassifier)

test_that("distances are correctly calculated with 10 data points(x,y coordinates) and 2 clusters", {
  X <- matrix(c(1,2,1,2,1,1,2,2,3,3,3,3,2,2,1,0,0,1,0,1), ncol = 2)
  C <- matrix(c(-1.6260988,0.302947,-0.5021358,0.7021099), ncol = 2)
  expected_distance <- matrix(c(19.16135007,25.41354767,13.15707847,19.40927607,9.152806869,7.148535269,13.40073287,15.40500447,21.65293047,23.65720207,5.766181796,8.160287796,2.170401596,4.564507596,0.5746213965,0.9788411965,3.372947196,2.968727396,7.767053196,7.362833396), ncol = 2)
  computed_distances <- fn.get_Xi_Ci_Distances(X = X, C = C)
  expect_equal(computed_distances, expected_distance)
  expect_equal(dim(computed_distances), dim(expected_distance))
  expect_that(computed_distances, is_a("matrix"))
})

test_that("membership is correct based on distances with 10 data points and 2 clusters", {
  distances <- matrix(c(19.16135007,25.41354767,13.15707847,19.40927607,9.152806869,7.148535269,13.40073287,15.40500447,21.65293047,23.65720207,5.766181796,8.160287796,2.170401596,4.564507596,0.5746213965,0.9788411965,3.372947196,2.968727396,7.767053196,7.362833396), ncol = 2)
  expected_membership <- matrix(c(0.4636236285,0.4690973216,0.4238405247,0.4450341736,0.3902656114,0.4149644339,0.4498170655,0.4323908072,0.4790290835,0.4664321796,0.7934872568,0.7842283983,0.8679666518,0.8266318925,0.9426386298,0.8865324619,0.8178423395,0.8508031046,0.767968906,0.7887093928), ncol = 2)
  computed_membership <- fn.get_membership(distances)
  expect_equal(computed_membership, expected_membership)
  expect_equal(dim(computed_membership), dim(expected_membership))
  expect_that(computed_membership, is_a("matrix"))
})

test_that("contributions are correct based on membership with 10 data points and 2 clusters", {
  membership <- matrix(c(0.4636236285,0.4690973216,0.4238405247,0.4450341736,0.3902656114,0.4149644339,0.4498170655,0.4323908072,0.4790290835,0.4664321796,0.7934872568,0.7842283983,0.8679666518,0.8266318925,0.9426386298,0.8865324619,0.8178423395,0.8508031046,0.767968906,0.7887093928), ncol = 2)
  expected_contributions <- matrix(c(0.3688009021,0.3742820515,0.3280989086,0.3499615075,0.292793435,0.318836284,0.3548406328,0.3369645096,0.3841458346,0.3716171863,0.6311990979,0.6257179485,0.6719010914,0.6500384925,0.707206565,0.681163716,0.6451593672,0.6630354904,0.6158541654,0.6283828137), ncol = 2)
  computed_contributions <- fn.contrib(membership)
  expect_equal(computed_contributions, expected_contributions)
  expect_equal(dim(computed_contributions), dim(expected_contributions))
  expect_that(computed_contributions, is_a("matrix"))
})

test_that("forward propagation (fn.amnfis_simulate) is working properly with 2 fixed clusters, fixed phi params and fixed data points(x,y coordinates)", {
  X <- matrix(c(1,2,1,2,1,1,2,2,3,3,3,3,2,2,1,0,0,1,0,1), ncol = 2)
  C <- matrix(c(-1.6260988,0.302947,-0.5021358,0.7021099), ncol = 2)
  PHI <- matrix(c(-0.004729057,0.285950071,1.172539,0.1799518), ncol = 2)
  phi_0 <- c(0.3773669,1.8119634)
  obj <- NULL
  obj$PHI <- PHI
  obj$phi_0 <- phi_0
  expected_prediction <- c(0.9568798251,0.9638655803,0.9271078922,0.9389457679,0.8872885183,0.8246005294,0.8413785165,0.901940737,0.8561568117,0.9138394971)
  pred <- fn.amnfis_simulate(obj = obj, X = X, C = C)
  expect_equal(pred, expected_prediction)
  expect_that(length(pred), equals(10))
  expect_that(pred, is_a("numeric"))
})
