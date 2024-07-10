###### <Table 2> Desciptive statistics and unit root test results of logarithm of VKOSPI ######
library(dplyr)
library(tidyr)
library(readr)
library(readxl)

setwd(dirname(rstudioapi::getSourceEditorContext()$path))

### Desciriptive statistics (Table 2)
data_vkospi <- as.data.frame(read_xlsx("./Data/data_raw_daily_use.xlsx"))[c('Date','vkospi')]
data_vkospi = data_vkospi[-1,]

data_vkospi[,1] = as.Date(as.integer(data_vkospi[,1]), origin = "1899-12-30")

data_vkospi = data_vkospi[-(1:2),]

library(moments)

mean(log(data_vkospi[,2]))
median(log(data_vkospi[,2]))
min((log(data_vkospi[,2])))
max((log(data_vkospi[,2])))
sd((log(data_vkospi[,2])))
skewness((log(data_vkospi[,2])))
kurtosis((log(data_vkospi[,2])))

### ADF test (Table 2)
library(urca)

y_adf1=ur.df(log(data_vkospi[,2]), type="drift") # with intercept
summary(y_adf1)

y_adf1@teststat
y_adf1@cval
y_adf1@lags

y_adf1_2=ur.df(log(data_vkospi[,2]), type="trend") # with intercept and linear trend
summary(y_adf1_2)

y_adf1_2@teststat
y_adf1_2@cval
y_adf1_2@lags

###### <Figure 1> Logarithm of VKOSPI from Jan 2016 to Mar 2023 ######
### VKOSPI plot (Figure 1)
plot(data_vkospi[,1], log(data_vkospi[,2]), ylab = "", xlab = "Date", type='l', col='blue')
title("Logarithm of Volatility KOSPI")