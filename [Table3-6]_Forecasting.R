############## Forecasting VKOSPI with Multi-Input LSTM ################
####### Table 3, Table 4 #######
library(tensorflow)
library(keras)
library(readxl)
library(readr)
library(recipes)
library(timetk)

setwd(dirname(rstudioapi::getSourceEditorContext()$path))
data <- as.data.frame(read_xlsx("./Data/data_raw_daily_use.xlsx")) %>% na.omit() %>% data.matrix()

tcode = data[1,]  # first element: Transformation code

data = data[-1,]
tdata = data[-(1:2),]
ncol(data)

#### Data Transformation
for (i in 2:ncol(data)){
  
  if(tcode[i] == 1){
    tdata[,i] <- data[-(1:2),i]
  }
  
  if(tcode[i] == 2){
    tdata[,i] <- diff(data[-1,i])
  }
  
  if(tcode[i] == 4){
    tdata[,i] <- log(data[-(1:2),i])
  } # log
  
  if(tcode[i] == 5){
    tdata[,i] <- diff(log(data[-1,i]))
  }
  
  if(tcode[i] == 6){
    tdata[,i] <- diff(diff(log(data[,i])))
  }
  
  if(tcode[i] == 7){
    tdata[,i] <- diff(data[-1,i]/data[1:(nrow(data)-1),i])
  }
}

head(tdata[,1])
tail(tdata[,1])

complete.cases(tdata)

row.names(tdata) <- tdata[,1]
data_use <- tdata[,2:ncol(tdata)]

library(reticulate)
library(tidyverse)

###### virtual environment connect
conda_list()

conda_list()[[1]][3] %>% 
  use_condaenv(required = TRUE)

###### LSTM (Table 3, Table 4)
### Random Seed
set.seed(1512)
set_random_seed(1512)

npred = 271

source("./functions/func-multilayer_lstm.R")

## LSTM
lstm_1 <- rolling.window.lstm(data_use, npred, 1, 1, 25, 40)
lstm_5 <- rolling.window.lstm(data_use, npred, 1, 5, 25, 40)
lstm_10 <- rolling.window.lstm(data_use, npred, 1, 10, 25, 40)
lstm_22 <- rolling.window.lstm(data_use, npred, 1, 22, 25, 40)

lstm_error_d <- cbind(lstm_1$errors, lstm_5$errors, lstm_10$errors, lstm_22$errors)

lstm_pred_d = matrix(NA,npred,4)
lstm_pred_d[,1] = lstm_1$pred
lstm_pred_d[-(1:4),2] = lstm_5$pred
lstm_pred_d[-(1:9),3] = lstm_10$pred
lstm_pred_d[-(1:21),4] = lstm_22$pred

## Multi-Input LSTM
multi_lstm_1 <- rolling.window.lstm.minput(data_use, npred, 1, 1, 25, 40)
multi_lstm_5 <- rolling.window.lstm.minput(data_use, npred, 1, 5, 25, 40)
multi_lstm_10 <- rolling.window.lstm.minput(data_use, npred, 1, 10, 25, 40)
multi_lstm_22 <- rolling.window.lstm.minput(data_use, npred, 1, 22, 25, 40)

multi_lstm_error_d <- cbind(multi_lstm_1$errors, multi_lstm_5$errors, multi_lstm_10$errors, multi_lstm_22$errors)

multi_lstm_pred_d = matrix(NA,npred,4)
multi_lstm_pred_d[,1] = multi_lstm_1$pred
multi_lstm_pred_d[-(1:4),2] = multi_lstm_5$pred
multi_lstm_pred_d[-(1:9),3] = multi_lstm_10$pred
multi_lstm_pred_d[-(1:21),4] = multi_lstm_22$pred

############ HAR (Table 3, Table 4) #############
library(readxl)
library(readr)
library(recipes)
library(timetk)
library(HDeconometrics)

setwd(dirname(rstudioapi::getSourceEditorContext()$path))
har_data <- as.data.frame(read_excel("./Data/data_raw_daily_har_use.xlsx")) %>% data.matrix()

har_data <- har_data[,2:ncol(har_data)] %>% na.omit()

npred = 271

source('./functions/func-ar_111.R')

har_1 <- ar.rolling.window2(har_data, npred, 1, 1, what = 'har')
har_5 <- ar.rolling.window2(har_data, npred, 1, 5, what = 'har')
har_10 <- ar.rolling.window2(har_data, npred, 1, 10, what = 'har')
har_22 <- ar.rolling.window2(har_data, npred, 1, 22, what = 'har')

har_d_error <- cbind(har_1$errors, har_5$errors, har_10$errors, har_22$errors)

har_d_pred = matrix(NA,npred,4)
har_d_pred[,1] = har_1$pred
har_d_pred[-(1:4),2] = har_5$pred
har_d_pred[-(1:9),3] = har_10$pred
har_d_pred[-(1:21),4] = har_22$pred

###### Forecasting VKOSPI except LSTM and HAR (Table 3, Table 4) ######
library(HDeconometrics)
library(tidyr)
library(readxl)
library(readr)
library(recipes)
library(timetk)

setwd(dirname(rstudioapi::getSourceEditorContext()$path))
data <- as.data.frame(read_xlsx("./Data/data_raw_daily_use.xlsx")) %>% data.matrix()

tcode = data[1,]  # first element: Transformation code

data = data[-1,]
tdata = data[-(1:2),]
ncol(data)

for (i in 2:ncol(data)){
  
  if(tcode[i] == 1){
    tdata[,i] <- data[-(1:2),i]
  }
  
  if(tcode[i] == 2){
    tdata[,i] <- diff(data[-1,i])
  }
  
  if(tcode[i] == 4){
    tdata[,i] <- log(data[-(1:2),i])
  } # log
  
  if(tcode[i] == 5){
    tdata[,i] <- diff(log(data[-1,i]))
  }
  
  if(tcode[i] == 6){
    tdata[,i] <- diff(diff(log(data[,i])))
  }
  
  if(tcode[i] == 7){
    tdata[,i] <- diff(data[-1,i]/data[1:(nrow(data)-1),i])
  }
}

head(tdata[,1])
tail(tdata[,1])

complete.cases(tdata)

row.names(tdata) <- tdata[,1]
data_use <- tdata[,2:ncol(tdata)]

colnames(data_use)

Y = data_use

normalize <- function(x) {
  return((x-min(x))/(max(x)-min(x)))
}

denormalize <- function(x, minval, maxval) {
  x*(maxval-minval) + minval
}

npred = 271

# LASSO
source("functions/func-lasso.R")
alpha <- 1

lasso_1 <- lasso.rolling.window(Y, npred, 1, 1, alpha, type = "lasso")
lasso_5 <- lasso.rolling.window(Y, npred, 1, 5, alpha, type = "lasso")
lasso_10 <- lasso.rolling.window(Y, npred, 1, 10, alpha, type = "lasso")
lasso_22 <- lasso.rolling.window(Y, npred, 1, 22, alpha, type = "lasso")

lasso_d_error <- cbind(lasso_1$errors, lasso_5$errors, lasso_10$errors, lasso_22$errors)

lasso_d_pred <- matrix(NA, npred, 4)
lasso_d_pred[, 1] <- lasso_1$pred
lasso_d_pred[-(1:4), 2] <- lasso_5$pred
lasso_d_pred[-(1:9), 3] <- lasso_10$pred
lasso_d_pred[-(1:21), 4] <- lasso_22$pred

# Adaptive LASSO
adalasso_1 <- lasso.rolling.window(Y, npred, 1, 1, alpha, type = "adalasso")
adalasso_5 <- lasso.rolling.window(Y, npred, 1, 5, alpha, type = "adalasso")
adalasso_10 <- lasso.rolling.window(Y, npred, 1, 10, alpha, type = "adalasso")
adalasso_22 <- lasso.rolling.window(Y, npred, 1, 22, alpha, type = "adalasso")

adalasso_d_error <- cbind(adalasso_1$errors, adalasso_5$errors, adalasso_10$errors, adalasso_22$errors)

adalasso_d_pred <- matrix(NA, npred, 4)
adalasso_d_pred[, 1] <- adalasso_1$pred
adalasso_d_pred[-(1:4), 2] <- adalasso_5$pred
adalasso_d_pred[-(1:9), 3] <- adalasso_10$pred
adalasso_d_pred[-(1:21), 4] <- adalasso_22$pred

# Elasticnet
alpha <- 0.5

elasticnet_1 <- lasso.rolling.window(Y, npred, 1, 1, alpha, type = "lasso")
elasticnet_5 <- lasso.rolling.window(Y, npred, 1, 5, alpha, type = "lasso")
elasticnet_10 <- lasso.rolling.window(Y, npred, 1, 10, alpha, type = "lasso")
elasticnet_22 <- lasso.rolling.window(Y, npred, 1, 22, alpha, type = "lasso")

elasticnet_d_error <- cbind(elasticnet_1$errors, elasticnet_5$errors, elasticnet_10$errors, elasticnet_22$errors)

elasticnet_d_pred <- matrix(NA, npred, 4)
elasticnet_d_pred[, 1] <- elasticnet_1$pred
elasticnet_d_pred[-(1:4), 2] <- elasticnet_5$pred
elasticnet_d_pred[-(1:9), 3] <- elasticnet_10$pred
elasticnet_d_pred[-(1:21), 4] <- elasticnet_22$pred

# Adaptive Elasticnet
adaelasticnet_1 <- lasso.rolling.window(Y, npred, 1, 1, alpha, type = "adalasso")
adaelasticnet_5 <- lasso.rolling.window(Y, npred, 1, 5, alpha, type = "adalasso")
adaelasticnet_10 <- lasso.rolling.window(Y, npred, 1, 10, alpha, type = "adalasso")
adaelasticnet_22 <- lasso.rolling.window(Y, npred, 1, 22, alpha, type = "adalasso")

adaelasticnet_d_error <- cbind(adaelasticnet_1$errors, adaelasticnet_5$errors, adaelasticnet_10$errors, adaelasticnet_22$errors)

adaelasticnet_d_pred <- matrix(NA, npred, 4)
adaelasticnet_d_pred[, 1] <- adaelasticnet_1$pred
adaelasticnet_d_pred[-(1:4), 2] <- adaelasticnet_5$pred
adaelasticnet_d_pred[-(1:9), 3] <- adaelasticnet_10$pred
adaelasticnet_d_pred[-(1:21), 4] <- adaelasticnet_22$pred

# Random Forest                 
source("functions/func-rf.R")
library(randomForest)

rf_1 <- rf.rolling.window(Y, npred, 1, 1)
rf_5 <- rf.rolling.window(Y, npred, 1, 5)
rf_10 <- rf.rolling.window(Y, npred, 1, 10)
rf_22 <- rf.rolling.window(Y, npred, 1, 22)

rf_d_error <- cbind(rf_1$errors, rf_5$errors, rf_10$errors, rf_22$errors)

rf_d_pred <- matrix(NA, npred, 4)
rf_d_pred[, 1] <- rf_1$pred
rf_d_pred[-(1:4), 2] <- rf_5$pred
rf_d_pred[-(1:9), 3] <- rf_10$pred
rf_d_pred[-(1:21), 4] <- rf_22$pred

# XGBoost
source('functions/func-xgb.R')
library(xgboost)

xgb_1 <- xgb.rolling.window(Y, npred, 1, 1)
xgb_5 <- xgb.rolling.window(Y, npred, 1, 5)
xgb_10 <- xgb.rolling.window(Y, npred, 1, 10)
xgb_22 <- xgb.rolling.window(Y, npred, 1, 22)

xgb_d_error <- cbind(xgb_1$errors, xgb_5$errors, xgb_10$errors, xgb_22$errors)

xgb_d_pred <- matrix(NA, npred, 4)
xgb_d_pred[, 1] <- xgb_1$pred
xgb_d_pred[-(1:4), 2] <- xgb_5$pred
xgb_d_pred[-(1:9), 3] <- xgb_10$pred
xgb_d_pred[-(1:21), 4] <- xgb_22$pred

# Neural Network
source('functions/func-nn.R')
library(h2o)
library(keras)

h2o.init()
nn_1 <- nn.rolling.window(Y, npred, 1, 1)
nn_5 <- nn.rolling.window(Y, npred, 1, 5)
nn_10 <- nn.rolling.window(Y, npred, 1, 10)
nn_22 <- nn.rolling.window(Y, npred, 1, 22)

nn_d_error <- cbind(nn_1$errors, nn_5$errors, nn_10$errors, nn_22$errors)

nn_d_pred <- matrix(NA, npred, 4)
nn_d_pred[, 1] <- nn_1$pred
nn_d_pred[-(1:4), 2] <- nn_5$pred
nn_d_pred[-(1:9), 3] <- nn_10$pred
nn_d_pred[-(1:21), 4] <- nn_22$pred


### Total Forecasting Results -> "./Data/Forecasting_Value_240306.xlsx"