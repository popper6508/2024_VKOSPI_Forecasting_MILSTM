########## Check Multi-Input LSTM ##########
########## Table7, Table8 ##########
library(tensorflow)
library(keras)
library(readxl)
library(readr)
library(recipes)
library(timetk)

setwd(dirname(rstudioapi::getSourceEditorContext()$path))

###### Multi -Input LSTM without input layer 2 and 3 (Table 7) #######
### Forecasting
data <- as.data.frame(read_xlsx("./Data/data_raw_daily_use.xlsx")) %>% na.omit() %>% data.matrix()

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

library(reticulate)
library(tidyverse)

conda_list()

conda_list()[[1]][3] %>% 
  use_condaenv(required = TRUE)

set.seed(100)
set_random_seed(100)

npred = 271

source("./functions/func-multilayer_lstm_test.R")

## 1 day ahead
lstm_test_1_1 <- rolling.window.lstm.minput.test(data_use, npred, 2, 1, 1, 25, 40) #### without layer 2
lstm_test_1_2 <- rolling.window.lstm.minput.test(data_use, npred, 3, 1, 1, 25, 40) #### without layer 3
lstm_test_1_3 <- rolling.window.lstm.minput.test(data_use, npred, 4, 1, 1, 25, 40) #### without layer 2 and 3

## 5 day ahead
lstm_test_5_1 <- rolling.window.lstm.minput.test(data_use, npred, 2, 1, 5, 25, 40) #### without layer 2
lstm_test_5_2 <- rolling.window.lstm.minput.test(data_use, npred, 3, 1, 5, 25, 40) #### without layer 3
lstm_test_5_3 <- rolling.window.lstm.minput.test(data_use, npred, 4, 1, 5, 25, 40) #### without layer 2 and 3

## 10 day ahead
lstm_test_10_1 <- rolling.window.lstm.minput.test(data_use, npred, 2, 1, 10, 25, 40) #### without layer 2
lstm_test_10_2 <- rolling.window.lstm.minput.test(data_use, npred, 3, 1, 10, 25, 40) #### without layer 3
lstm_test_10_3 <- rolling.window.lstm.minput.test(data_use, npred, 4, 1, 10, 25, 40) #### without layer 2 and 3

## 22 day ahead
lstm_test_22_1 <- rolling.window.lstm.minput.test(data_use, npred, 2, 1, 22, 25, 40) #### without layer 2
lstm_test_22_2 <- rolling.window.lstm.minput.test(data_use, npred, 3, 1, 22, 25, 40) #### without layer 3
lstm_test_22_3 <- rolling.window.lstm.minput.test(data_use, npred, 4, 1, 22, 25, 40) #### without layer 2 and 3

### Save forecasting results
lstm_pred_d_testvar = matrix(NA,npred,12)
lstm_pred_d_testvar[,1] = lstm_test_1_1$pred
lstm_pred_d_testvar[,2] = lstm_test_1_2$pred
lstm_pred_d_testvar[,3] = lstm_test_1_3$pred

lstm_pred_d_testvar[-(1:4),4] = lstm_test_5_1$pred
lstm_pred_d_testvar[-(1:4),5] = lstm_test_5_2$pred
lstm_pred_d_testvar[-(1:4),6] = lstm_test_5_3$pred

lstm_pred_d_testvar[-(1:9),7] = lstm_test_10_1$pred
lstm_pred_d_testvar[-(1:9),8] = lstm_test_10_2$pred
lstm_pred_d_testvar[-(1:9),9] = lstm_test_10_3$pred

lstm_pred_d_testvar[-(1:21),10] = lstm_test_22_1$pred
lstm_pred_d_testvar[-(1:21),11] = lstm_test_22_2$pred
lstm_pred_d_testvar[-(1:21),12] = lstm_test_22_3$pred

lstmday1 = rbind(lstm_test_1_1$errors, lstm_test_1_2$errors, lstm_test_1_3$errors)
lstmday5 = rbind(lstm_test_5_1$errors, lstm_test_5_2$errors, lstm_test_5_3$errors)
lstmday10 = rbind(lstm_test_10_1$errors,lstm_test_10_2$errors,lstm_test_10_3$errors)
lstmday22 = rbind(lstm_test_22_1$errors,lstm_test_22_2$errors,lstm_test_22_3$errors)

total_error = cbind(lstmday1, lstmday5, lstmday10, lstmday22)
row.names(total_error) = c("Input Layer 1,3,4", "Input Layer 1,2,4", "Input Layer 1,4")

save.image("./Result_Record/[Table7]_Variable_Test_Using_MILSTM.RData")

####### Comparison between Multi_Input LSTM model and the model by Li et al.(2023) (Table 8) ######
### Forecasting
data <- as.data.frame(read_xlsx("./Data/data_raw_daily_use.xlsx")) %>% na.omit() %>% data.matrix()

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

library(reticulate)
library(tidyverse)

conda_list()

conda_list()[[1]][3] %>% 
  use_condaenv(required = TRUE)

set.seed(100)
set_random_seed(100)

npred = 271

source("./functions/func-multilayer_lstm_test.R")

m.lstm_1 <- rolling.window.lstm.minput.o(data_use, npred, 1, 1, 25, 40)
m.lstm_5 <- rolling.window.lstm.minput.o(data_use, npred, 1, 5, 25, 40)
m.lstm_10 <- rolling.window.lstm.minput.o(data_use, npred, 1, 10, 25, 40)
m.lstm_22 <- rolling.window.lstm.minput.o(data_use, npred, 1, 22, 25, 40)

m.lstm_pred_d = matrix(NA,npred,4)
m.lstm_pred_d[,1] = m.lstm_1$pred
m.lstm_pred_d[-(1:4),2] = m.lstm_5$pred
m.lstm_pred_d[-(1:9),3] = m.lstm_10$pred
m.lstm_pred_d[-(1:21),4] = m.lstm_22$pred

save.image("./Result_Record/[Table8]_Lstm_Li_Ver_Result.RData")

### GW Test
library(readxl)
library(readr)
library(recipes)
library(timetk)

setwd(dirname(rstudioapi::getSourceEditorContext()$path))

load("./Result_Record/[Table8]_Lstm_Li_Ver_Result.RData")

prediction_result = read_xlsx("./Data/Forecasting_Value_240306.xlsx") %>% as.data.frame()
prediction_result_mlstm = prediction_result[,c(5, 15, 25, 35)]

real = tail(prediction_result[,2], 271)

source("./functions/gwtest.R")
library(sandwich)
library(dplyr)
library(stringr)

gw_1_lstm_com = gw.test(prediction_result_mlstm[,1], m.lstm_pred_d[,1], real, tau=1, T=271, method="NeweyWest", alternative="two.sided")

gw_5_lstm_com = gw.test(tail(prediction_result_mlstm[,2], 267), tail(m.lstm_pred_d[,2], 267), tail(real, 267), tau=5, T=271, method="NeweyWest", alternative="two.sided")

gw_10_lstm_com = gw.test(tail(prediction_result_mlstm[,3], 262), tail(m.lstm_pred_d[,3], 262), tail(real, 262), tau=10, T=271, method="NeweyWest", alternative="two.sided")

gw_22_lstm_com = gw.test(tail(prediction_result_mlstm[,4], 250), tail(m.lstm_pred_d[,4], 250), tail(real, 250), tau=22, T=271, method="NeweyWest", alternative="two.sided")

save.image("./Result_Record/[Table8]_MILSTM_Compare_Error_and_GWTest.RData")
