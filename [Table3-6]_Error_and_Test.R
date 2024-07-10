######## Error Table / GW test / MCS Test using total forecasting results ########
#### Table3, Table4, Table5, Table6 ####
library(readxl)
library(readr)
library(recipes)
library(timetk)

setwd(dirname(rstudioapi::getSourceEditorContext()$path))
prediction_result = read_xlsx("./Data/Forecasting_Value_240306.xlsx") %>% data.frame()

#### Error Value Correctness Check: RMSE / MAE (Table 3, Table 4) ####
PRED_1 = prediction_result[,3:12]
PRED_5 = prediction_result[,13:22]
PRED_10 = prediction_result[,23:32]
PRED_22 = prediction_result[,33:42]

real = prediction_result[,2]

REAL=cbind(real, real, real, real, real, real, real, real, real, real)

LOSS_1=PRED_1-REAL
LOSS_5=PRED_5-REAL
LOSS_10=PRED_10-REAL
LOSS_22=PRED_22-REAL

LOSS1_1=LOSS_1^2
LOSS1_5=LOSS_5^2
LOSS1_10=LOSS_10^2
LOSS1_22=LOSS_22^2

LOSS2_1=abs(LOSS_1)
LOSS2_5=abs(LOSS_5)
LOSS2_10=abs(LOSS_10)
LOSS2_22=abs(LOSS_22)

RMSE = matrix(NA, nrow = 10, ncol = 4)
MAE = matrix(NA, nrow = 10, ncol = 4)

colnames(MAE) = c("1day","5day","10day","22day")
colnames(RMSE) = c("1day","5day","10day","22day")

row.names(MAE) = c("HAR", "LSTM", "Multi_LSTM", "LASSO", "AdaLASSO", "ElasticNet",
                   "AdaElasticNet", "RF", "XGBoost", "NN")
row.names(RMSE) = c("HAR", "LSTM", "Multi_LSTM", "LASSO", "AdaLASSO", "ElasticNet",
                    "AdaElasticNet", "RF", "XGBoost", "NN")

for (i in 1:10) {
  RMSE[i,1]=sqrt(mean(LOSS1_1[,i]))
  RMSE[i,2]=sqrt(mean(na.omit(LOSS1_5[,i])))
  RMSE[i,3]=sqrt(mean(na.omit(LOSS1_10[,i])))
  RMSE[i,4]=sqrt(mean(na.omit(LOSS1_22[,i])))
  
  MAE[i,1]=mean(abs(LOSS2_1[,i]))
  MAE[i,2]=mean(abs(na.omit(LOSS2_5[,i])))
  MAE[i,3]=mean(abs(na.omit(LOSS2_10[,i])))
  MAE[i,4]=mean(abs(na.omit(LOSS2_22[,i])))
}

#### GW Test : with HAR (Table 5) ####
source("./functions/gwtest.R")
library(sandwich)
library(dplyr)
library(stringr)

gw_test_pvalue = matrix(NA, nrow = 9, ncol = 4)
gw_test_statistics = matrix(NA, nrow = 9, ncol = 4)

colnames(gw_test_pvalue) = c("1day","5day","10day","22day")
row.names(gw_test_pvalue) = c("LSTM", "Multi_LSTM", "LASSO", "AdaLASSO", "ElasticNet",
                              "AdaElasticNet", "RF", "XGBoost", "NN")
colnames(gw_test_statistics) = c("1day","5day","10day","22day")
row.names(gw_test_statistics) = c("LSTM", "Multi_LSTM", "LASSO", "AdaLASSO", "ElasticNet",
                                  "AdaElasticNet", "RF", "XGBoost", "NN")

## Horizon 1
gw_1_lstm = gw.test(PRED_1[,1], PRED_1[,2], real, tau=1, T=271, method="NeweyWest", alternative="two.sided")
gw_1_multi = gw.test(PRED_1[,1], PRED_1[,3], real, tau=1, T=271, method="NeweyWest", alternative="two.sided")
gw_1_lasso = gw.test(PRED_1[,1], PRED_1[,4], real, tau=1, T=271, method="NeweyWest", alternative="two.sided")
gw_1_adalasso = gw.test(PRED_1[,1], PRED_1[,5], real, tau=1, T=271, method="NeweyWest", alternative="two.sided")
gw_1_ela = gw.test(PRED_1[,1], PRED_1[,6], real, tau=1, T=271, method="NeweyWest", alternative="two.sided")
gw_1_adaela = gw.test(PRED_1[,1], PRED_1[,7], real, tau=1, T=271, method="NeweyWest", alternative="two.sided")
gw_1_rf = gw.test(PRED_1[,1], PRED_1[,8], real, tau=1, T=271, method="NeweyWest", alternative="two.sided")
gw_1_xgb = gw.test(PRED_1[,1], PRED_1[,9], real, tau=1, T=271, method="NeweyWest", alternative="two.sided")
gw_1_nn = gw.test(PRED_1[,1], PRED_1[,10], real, tau=1, T=271, method="NeweyWest", alternative="two.sided")

gw_test_pvalue[,1] <- c(gw_1_lstm$p.value, gw_1_multi$p.value, gw_1_lasso$p.value,
                        gw_1_adalasso$p.value, gw_1_ela$p.value, gw_1_adaela$p.value, gw_1_rf$p.value,
                        gw_1_xgb$p.value, gw_1_nn$p.value)
gw_test_statistics[,1] <- c(gw_1_lstm$statistic, gw_1_multi$statistic, gw_1_lasso$statistic,
                            gw_1_adalasso$statistic, gw_1_ela$statistic, gw_1_adaela$statistic, gw_1_rf$statistic,
                            gw_1_xgb$statistic, gw_1_nn$statistic)

## Horizon 5
gw_5_lstm = gw.test(tail(PRED_5[,1],267), tail(PRED_5[,2],267), tail(real,267), tau=5, T=271, method="NeweyWest", alternative="two.sided")
gw_5_multi = gw.test(tail(PRED_5[,1],267), tail(PRED_5[,3],267), tail(real,267), tau=5, T=271, method="NeweyWest", alternative="two.sided")
gw_5_lasso = gw.test(tail(PRED_5[,1],267), tail(PRED_5[,4],267), tail(real,267), tau=5, T=271, method="NeweyWest", alternative="two.sided")
gw_5_adalasso = gw.test(tail(PRED_5[,1],267), tail(PRED_5[,5],267), tail(real,267), tau=5, T=271, method="NeweyWest", alternative="two.sided")
gw_5_ela = gw.test(tail(PRED_5[,1],267), tail(PRED_5[,6],267), tail(real,267), tau=5, T=271, method="NeweyWest", alternative="two.sided")
gw_5_adaela = gw.test(tail(PRED_5[,1],267), tail(PRED_5[,7],267), tail(real,267), tau=5, T=271, method="NeweyWest", alternative="two.sided")
gw_5_rf = gw.test(tail(PRED_5[,1],267), tail(PRED_5[,8],267), tail(real,267), tau=5, T=271, method="NeweyWest", alternative="two.sided")
gw_5_xgb = gw.test(tail(PRED_5[,1],267), tail(PRED_5[,9],267), tail(real,267), tau=5, T=271, method="NeweyWest", alternative="two.sided")
gw_5_nn = gw.test(tail(PRED_5[,1],267), tail(PRED_5[,10],267), tail(real,267), tau=5, T=271, method="NeweyWest", alternative="two.sided")

gw_test_pvalue[,2] <- c(gw_5_lstm$p.value, gw_5_multi$p.value, gw_5_lasso$p.value,
                        gw_5_adalasso$p.value, gw_5_ela$p.value, gw_5_adaela$p.value, gw_5_rf$p.value,
                        gw_5_xgb$p.value, gw_5_nn$p.value)
gw_test_statistics[,2] <- c(gw_5_lstm$statistic, gw_5_multi$statistic, gw_5_lasso$statistic,
                            gw_5_adalasso$statistic, gw_5_ela$statistic, gw_5_adaela$statistic, gw_5_rf$statistic,
                            gw_5_xgb$statistic, gw_5_nn$statistic)

## Horizon 10
gw_10_lstm = gw.test(tail(PRED_10[,1],262), tail(PRED_10[,2],262), tail(real,262), tau=10, T=271, method="NeweyWest", alternative="two.sided")
gw_10_multi = gw.test(tail(PRED_10[,1],262), tail(PRED_10[,3],262), tail(real,262), tau=10, T=271, method="NeweyWest", alternative="two.sided")
gw_10_lasso = gw.test(tail(PRED_10[,1],262), tail(PRED_10[,4],262), tail(real,262), tau=10, T=271, method="NeweyWest", alternative="two.sided")
gw_10_adalasso = gw.test(tail(PRED_10[,1],262), tail(PRED_10[,5],262), tail(real,262), tau=10, T=271, method="NeweyWest", alternative="two.sided")
gw_10_ela = gw.test(tail(PRED_10[,1],262), tail(PRED_10[,6],262), tail(real,262), tau=10, T=271, method="NeweyWest", alternative="two.sided")
gw_10_adaela = gw.test(tail(PRED_10[,1],262), tail(PRED_10[,7],262), tail(real,262), tau=10, T=271, method="NeweyWest", alternative="two.sided")
gw_10_rf = gw.test(tail(PRED_10[,1],262), tail(PRED_10[,8],262), tail(real,262), tau=10, T=271, method="NeweyWest", alternative="two.sided")
gw_10_xgb = gw.test(tail(PRED_10[,1],262), tail(PRED_10[,9],262), tail(real,262), tau=10, T=271, method="NeweyWest", alternative="two.sided")
gw_10_nn = gw.test(tail(PRED_10[,1],262), tail(PRED_10[,10],262), tail(real,262), tau=10, T=271, method="NeweyWest", alternative="two.sided")

gw_test_pvalue[,3] <- c(gw_10_lstm$p.value, gw_10_multi$p.value, gw_10_lasso$p.value,
                        gw_10_adalasso$p.value, gw_10_ela$p.value, gw_10_adaela$p.value, gw_10_rf$p.value,
                        gw_10_xgb$p.value, gw_10_nn$p.value)
gw_test_statistics[,3] <- c(gw_10_lstm$statistic, gw_10_multi$statistic, gw_10_lasso$statistic,
                            gw_10_adalasso$statistic, gw_10_ela$statistic, gw_10_adaela$statistic, gw_10_rf$statistic,
                            gw_10_xgb$statistic, gw_10_nn$statistic)

## Horizon 22
gw_22_lstm = gw.test(tail(PRED_22[,1],250), tail(PRED_22[,2],250), tail(real,250), tau=22, T=271, method="NeweyWest", alternative="two.sided")
gw_22_multi = gw.test(tail(PRED_22[,1],250), tail(PRED_22[,3],250), tail(real,250), tau=22, T=271, method="NeweyWest", alternative="two.sided")
gw_22_lasso = gw.test(tail(PRED_22[,1],250), tail(PRED_22[,4],250), tail(real,250), tau=22, T=271, method="NeweyWest", alternative="two.sided")
gw_22_adalasso = gw.test(tail(PRED_22[,1],250), tail(PRED_22[,5],250), tail(real,250), tau=22, T=271, method="NeweyWest", alternative="two.sided")
gw_22_ela = gw.test(tail(PRED_22[,1],250), tail(PRED_22[,6],250), tail(real,250), tau=22, T=271, method="NeweyWest", alternative="two.sided")
gw_22_adaela = gw.test(tail(PRED_22[,1],250), tail(PRED_22[,7],250), tail(real,250), tau=22, T=271, method="NeweyWest", alternative="two.sided")
gw_22_rf = gw.test(tail(PRED_22[,1],250), tail(PRED_22[,8],250), tail(real,250), tau=22, T=271, method="NeweyWest", alternative="two.sided")
gw_22_xgb = gw.test(tail(PRED_22[,1],250), tail(PRED_22[,9],250), tail(real,250), tau=22, T=271, method="NeweyWest", alternative="two.sided")
gw_22_nn = gw.test(tail(PRED_22[,1],250), tail(PRED_22[,10],250), tail(real,250), tau=22, T=271, method="NeweyWest", alternative="two.sided")

gw_test_pvalue[,4] <- c(gw_22_lstm$p.value, gw_22_multi$p.value, gw_22_lasso$p.value,
                        gw_22_adalasso$p.value, gw_22_ela$p.value, gw_22_adaela$p.value, gw_22_rf$p.value,
                        gw_22_xgb$p.value, gw_22_nn$p.value)
gw_test_statistics[,4] <- c(gw_22_lstm$statistic, gw_22_multi$statistic, gw_22_lasso$statistic,
                            gw_22_adalasso$statistic, gw_22_ela$statistic, gw_22_adaela$statistic, gw_22_rf$statistic,
                            gw_22_xgb$statistic, gw_22_nn$statistic)

#### MCS Test (Table 6) ####
library(MCS)

MCS_RMSE2_1 <- MCSprocedure(LOSS1_1, alpha=0.5, B=5000, statistic="Tmax")
MCS_RMSE2_5 <- MCSprocedure(na.omit(LOSS1_5), alpha=0.5, B=5000, statistic="Tmax")
MCS_RMSE2_10 <- MCSprocedure(na.omit(LOSS1_10), alpha=0.5, B=5000, statistic="Tmax")
MCS_RMSE2_22 <- MCSprocedure(na.omit(LOSS1_22), alpha=0.5, B=5000, statistic="Tmax")

MCS_MAE2_1 <- MCSprocedure(LOSS2_1, alpha=0.5, B=5000, statistic="Tmax")
MCS_MAE2_5 <- MCSprocedure(na.omit(LOSS2_5), alpha=0.5, B=5000, statistic="Tmax")
MCS_MAE2_10 <- MCSprocedure(na.omit(LOSS2_10), alpha=0.5, B=5000, statistic="Tmax")
MCS_MAE2_22 <- MCSprocedure(na.omit(LOSS2_22), alpha=0.5, B=5000, statistic="Tmax")

save.image("./Result_Record/[Table3_6]_Error_and_Test.RData")