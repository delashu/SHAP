#R version 3.6.1
##################### Load necessary libraries #####################
library(xgboost)
library(tidyverse)
library(Matrix)
library(data.table)
library(rsample)


##################### Manually load in functions #####################
#load(url("https://github.com/pablo14/shap-values/blob/master/shap.R"))

#
#**                             
#**         FUNCTIONS           
#**                              
#

plot.shap.summary <- function(data_long){
  x_bound <- max(abs(data_long$value))
  require('ggforce') # for `geom_sina`
  plot1 <- ggplot(data = data_long)+
    coord_flip() + 
    # sina plot: 
    geom_sina(aes(x = variable, y = value, color = stdfvalue)) +
    # print the mean absolute value: 
    geom_text(data = unique(data_long[, c("variable", "mean_value"), with = F]),
              aes(x = variable, y=-Inf, label = sprintf("%.3f", mean_value)),
              size = 3, alpha = 0.7,
              hjust = -0.2, 
              fontface = "bold") + # bold
    # # add a "SHAP" bar notation
    # annotate("text", x = -Inf, y = -Inf, vjust = -0.2, hjust = 0, size = 3,
    #          label = expression(group("|", bar(SHAP), "|"))) + 
    scale_color_gradient(low="#FFCC33", high="#6600CC", 
                         breaks=c(0,1), labels=c("Low","High")) +
    theme_bw() + 
    theme(axis.line.y = element_blank(), axis.ticks.y = element_blank(), # remove axis line
          legend.position="bottom") + 
    geom_hline(yintercept = 0) + # the vertical line
    scale_y_continuous(limits = c(-x_bound, x_bound)) +
    # reverse the order of features
    scale_x_discrete(limits = rev(levels(data_long$variable)) 
    ) + 
    labs(y = "SHAP value (impact on model output)", x = "", color = "Feature value") 
  return(plot1)
}

shap.score.rank <- function(xgb_model = xgb_mod, shap_approx = TRUE, 
                            X_train = mydata$train_mm){
  require(xgboost)
  require(data.table)
  shap_contrib <- predict(xgb_model, X_train,
                          predcontrib = TRUE, approxcontrib = shap_approx)
  shap_contrib <- as.data.table(shap_contrib)
  shap_contrib[,BIAS:=NULL]
  cat('make SHAP score by decreasing order\n\n')
  mean_shap_score <- colMeans(abs(shap_contrib))[order(colMeans(abs(shap_contrib)), decreasing = T)]
  return(list(shap_score = shap_contrib,
              mean_shap_score = (mean_shap_score)))
}
std1 <- function(x){
  return ((x - min(x, na.rm = T))/(max(x, na.rm = T) - min(x, na.rm = T)))
}


# prep shap data
shap.prep <- function(shap  = shap_result, X_train = mydata$train_mm, top_n){
  require(ggforce)
  # descending order
  if (missing(top_n)) top_n <- dim(X_train)[2] # by default, use all features
  if (!top_n%in%c(1:dim(X_train)[2])) stop('supply correct top_n')
  require(data.table)
  shap_score_sub <- as.data.table(shap$shap_score)
  shap_score_sub <- shap_score_sub[, names(shap$mean_shap_score)[1:top_n], with = F]
  shap_score_long <- melt.data.table(shap_score_sub, measure.vars = colnames(shap_score_sub))
  
  # feature values: the values in the original dataset
  fv_sub <- as.data.table(as.matrix(X_train))[, names(shap$mean_shap_score)[1:top_n], with = F]
  # standardize feature values
  fv_sub_long <- melt.data.table(fv_sub, measure.vars = colnames(fv_sub))
  fv_sub_long[, stdfvalue := std1(value), by = "variable"]
  # SHAP value: value
  # raw feature value: rfvalue; 
  # standarized: stdfvalue
  names(fv_sub_long) <- c("variable", "rfvalue", "stdfvalue" )
  shap_long2 <- cbind(shap_score_long, fv_sub_long[,c('rfvalue','stdfvalue')])
  shap_long2[, mean_value := mean(abs(value)), by = variable]
  setkey(shap_long2, variable)
  return(shap_long2) 
}

#
#
#*
#*            END FUNCTIONS
#*
#
#


##################### Read in data #####################
mydata <- readRDS(file="/location/of/file.rds")

#split the data into train and testing
set.seed(312)

#use the 'rsample' package to prepare for testing and training split
#replace "outcome" as the column name of your outcome in your data
first_split <- initial_split(mydata,prop = 8/10, strata = "outcome") 

#create training and testing data
ptrain <- training(first_split)    #80% training
ptest <- testing(first_split)      #20% testing/validation

rownames(ptrain) <- NULL
rownames(ptest) <- NULL

#Copy the training data to a new data object
ptrain_lab <- ptrain
#Remove the outcome variable 
ptrain_lab$outcome <- NULL

#change all chr to factor with an 'unclass' statement.
data_train <- as.data.frame(unclass(ptrain_lab))

#turn the data into a matrix
data_train_mat <- as.matrix(data_train, nrow = nrow(data_train), ncol = ncol(data_train))

#create a sparse matrix
sparse_matrix <- Matrix(data_train_mat, sparse = TRUE)
train_matrix <- xgb.DMatrix(data = sparse_matrix, label = ptrain$pc)

#Gradient Boosting Machine (any GBM will do)
gbm_model <- xgboost(data = train_matrix, max.depth = 2, 
                     eta = 1, nthread = 2, nrounds = 2, 
                     objective = "count:poisson", verbose = 2)

#copy the test data to a new object
ptest_lab <- ptest

#remove the outcome variable from the copied data
ptest_lab$pc <- NULL

#change character variables to factors with 'unclass'
data_test <- as.data.frame(unclass(ptest_lab))

#create a sparse matrix
data_test_mat <- as.matrix(data_test, nrow = nrow(data_test), ncol = ncol(data_test))
sparse_matrix_test <- Matrix(data_test_mat, sparse = TRUE)

#GBM predict on the test dataset
pred <- predict(gbm_model, sparse_matrix_test, shap)

#Create the SHAP scores 
shap_scores <- shap.score.rank(xgb_model = gbm_model,
                               shap_approx = TRUE,
                               X_train=sparse_matrix)

#Output the Global top 10 important variables into a dataframe
shap_scores_std <- shap.prep(shap  = shap_scores, 
                             X_train = sparse_matrix, 
                             top_n = 10)

#Plot the top 10 variables in a SHAP impact plot
plot.shap.summary(shap_scores_std)


xgb.plot.shap(data = sparse_matrix, shap_contrib = shap_scores$shap_score, model = bst)
