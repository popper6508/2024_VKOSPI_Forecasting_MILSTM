########################################
###### Multi Layer LSTM Functions ######
########################################

normalize <- function(x) {
  return((x-min(x))/(max(x)-min(x)))
}

denormalize <- function(x, minval, maxval) {
  x*(maxval-minval) + minval
}

###### LSTM ######
run_single_lstm=function(Y,indice,lag, batch_size = 30, unit_n = 32){
  horizon = lag
  Y2 = Y %>% as.data.frame()
  Y3 = lapply(Y2, normalize) %>% as.data.frame() %>% as.matrix()
  
  ### Old version code ###
  # aux=embed(Y3,4+lag)
  # y=aux[,indice]
  # X=aux[,-c(1:(ncol(Y3)*lag))]  
  # 
  # if(lag==1){
  #   X.out=tail(aux,1)[1:ncol(X)]  
  # }else{
  #   X.out=aux[,-c(1:(ncol(Y3)*(lag-1)))]
  #   X.out=tail(X.out,1)[1:ncol(X)]
  # }
  
  X=embed(as.matrix(Y3),4)
  
  Xin=X[-c((nrow(X)-horizon+1):nrow(X)),]
  Xout=X[nrow(X),]
  
  y=tail(Y3[,1],nrow(Xin))
  X = Xin 
  X.out = Xout
  
  ###
  X2 <- X %>% replace(!0, 0) 
  
  for(i in 0:(ncol(Y3)-1)){
    X2[,(4*i+4)] <- X[,(i+1)]
    X2[,(4*i+3)] <- X[,(i+ncol(Y3)+1)]
    X2[,(4*i+2)] <- X[,(i+2*ncol(Y3)+1)]
    X2[,(4*i+1)] <- X[,(i+3*ncol(Y3)+1)]
  }
  
  X.out2 <- X.out %>% replace(!0, 0)
  
  for(i in 0:(ncol(Y3)-1)){
    X.out2[(4*i+4)] <- X.out[(i+1)]
    X.out2[(4*i+3)] <- X.out[(i+ncol(Y3)+1)]
    X.out2[(4*i+2)] <- X.out[(i+2*ncol(Y3)+1)]
    X.out2[(4*i+1)] <- X.out[(i+3*ncol(Y3)+1)]
  }
  
  ###  
  X.arr = array(
    data = as.numeric(unlist(X2)),
    dim = c(nrow(X), 4, ncol(Y3)))
  
  X.out.arr = array(
    data = as.numeric(unlist(X.out2)),
    dim = c(1, 4, ncol(Y3)))
  
  batch_size = batch_size
  unit_n = unit_n
  feature = ncol(Y3)
  epochs = 100
  
  model = keras_model_sequential()
  
  model %>% layer_lstm(units = unit_n, 
                       input_shape = c(4, feature),
                       stateful = FALSE) %>%
    layer_dense(units = 1) 
  
  model %>% compile(loss = 'mse', optimizer = 'adam')
  
  model %>% summary()
  
  history = model %>% fit(X.arr, y, epochs = epochs,
                          batch_size = batch_size, shuffle = FALSE, verbose = 2)
  
  pred = model(X.out.arr) %>% denormalize(min(Y2[,indice]),max(Y2[,indice])) # De-normalization
  
  return(list("model"=model,"pred"=pred))
}

run_multi_lstm=function(Y,indice,lag, batch_size = 30, unit_n = 32){
  horizon = lag
  Y2 = Y %>% as.data.frame()
  Y3 = lapply(Y, normalize) %>% as.data.frame() %>% as.matrix()
  
  ### Old version code ###
  # aux=embed(Y3,4+lag)
  # y=aux[,indice]
  # X=aux[,-c(1:(ncol(Y3)*lag))]  
  # 
  # if(lag==1){
  #   X.out=tail(aux,1)[1:ncol(X)]  
  # }else{
  #   X.out=aux[,-c(1:(ncol(Y3)*(lag-1)))]
  #   X.out=tail(X.out,1)[1:ncol(X)]
  # }
  
  X=embed(as.matrix(Y3),4)
  
  Xin=X[-c((nrow(X)-horizon+1):nrow(X)),]
  Xout=X[nrow(X),]
  
  y=tail(Y3[,1],nrow(Xin))
  X = Xin 
  X.out = Xout
  
  ###
  X2 <- X %>% replace(!0, 0) 
  
  for(i in 0:(ncol(Y3)-1)){
    X2[,(4*i+4)] <- X[,(i+1)]
    X2[,(4*i+3)] <- X[,(i+ncol(Y3)+1)]
    X2[,(4*i+2)] <- X[,(i+2*ncol(Y3)+1)]
    X2[,(4*i+1)] <- X[,(i+3*ncol(Y3)+1)]
  }
  
  X.out2 <- X.out %>% replace(!0, 0)
  
  for(i in 0:(ncol(Y3)-1)){
    X.out2[(4*i+4)] <- X.out[(i+1)]
    X.out2[(4*i+3)] <- X.out[(i+ncol(Y3)+1)]
    X.out2[(4*i+2)] <- X.out[(i+2*ncol(Y3)+1)]
    X.out2[(4*i+1)] <- X.out[(i+3*ncol(Y3)+1)]
  }
  
  ###  
  X.arr = array(
    data = as.numeric(unlist(X2)),
    dim = c(nrow(X), 4, ncol(Y3)))
  
  X.out.arr = array(
    data = as.numeric(unlist(X.out2)),
    dim = c(1, 4, ncol(Y3)))
  
  batch_size = batch_size
  unit_n = unit_n
  feature = ncol(Y3)
  epochs = 100
  
  model = keras_model_sequential()
  
  model %>%
    layer_lstm(units = unit_n, input_shape = c(4, feature), stateful = FALSE, return_sequences = TRUE) %>%
    layer_lstm(units = unit_n, return_sequences = TRUE) %>%
    layer_lstm(units = unit_n, return_sequences = TRUE) %>%
    layer_lstm(units = unit_n, return_sequences = FALSE) %>%
    layer_dense(units = 1)
  
  model %>% compile(loss = 'mse', optimizer = 'adam')
  
  model %>% summary()
  
  history = model %>% fit(X.arr, y, epochs = epochs,
                          batch_size = batch_size, shuffle = FALSE, verbose = 2)
  
  pred = model(X.out.arr) %>% denormalize(min(Y2[,indice]),max(Y2[,indice])) # De-normalization
  
  return(list("model"=model,"pred"=pred))
}

###### Multi-Input LSTM ######
run_multiinput_lstm=function(Y,indice,lag, batch_size = 30, unit_n = 32){
  horizon = lag
  Y2 = Y %>% as.data.frame()
  Y3 = lapply(Y2, normalize) %>% as.data.frame() %>% as.matrix()
  
  # aux=embed(Y3,4+lag)
  # y=aux[,indice]
  # X=aux[,-c(1:(ncol(Y3)*lag))]
  # 
  # if(lag==1){
  #   X.out=tail(aux,1)[1:ncol(X)]  
  # }else{
  #   X.out=aux[,-c(1:(ncol(Y3)*(lag-1)))]
  #   X.out=tail(X.out,1)[1:ncol(X)]
  # }
  
  X=embed(as.matrix(Y3),4)
  
  Xin=X[-c((nrow(X)-horizon+1):nrow(X)),]
  Xout=X[nrow(X),]
  
  y=tail(Y3[,1],nrow(Xin))
  X = Xin 
  X.out = Xout
  
  ###
  X2_1 <- X[,1:44] %>% replace(!0, 0) 
  
  for(i in 1:11){
    X2_1[,(4*i)] <- X[,(i+1)]
    X2_1[,(4*i-1)] <- X[,(i+ncol(Y3)+1)]
    X2_1[,(4*i-2)] <- X[,(i+2*ncol(Y3)+1)]
    X2_1[,(4*i-3)] <- X[,(i+3*ncol(Y3)+1)]
  }
  
  X2_2 <- X[,1:36] %>% replace(!0, 0) 
  
  for(i in 12:20){
    X2_2[,(4*i-44)] <- X[,(i+1)]
    X2_2[,(4*i-45)] <- X[,(i+ncol(Y3)+1)]
    X2_2[,(4*i-46)] <- X[,(i+2*ncol(Y3)+1)]
    X2_2[,(4*i-47)] <- X[,(i+3*ncol(Y3)+1)]
  }
  
  X2_3 <- X[,1:124] %>% replace(!0, 0) 
  
  for(i in 21:51){
    X2_3[,(4*i-80)] <- X[,(i+1)]
    X2_3[,(4*i-81)] <- X[,(i+ncol(Y3)+1)]
    X2_3[,(4*i-82)] <- X[,(i+2*ncol(Y3)+1)]
    X2_3[,(4*i-83)] <- X[,(i+3*ncol(Y3)+1)]
  }
  
  X2_4 <- X[,1:12] %>% replace(!0, 0) 
  
  for(i in 52:54){
    X2_4[,(4*i-204)] <- X[,(i+1)]
    X2_4[,(4*i-205)] <- X[,(i+ncol(Y3)+1)]
    X2_4[,(4*i-206)] <- X[,(i+2*ncol(Y3)+1)]
    X2_4[,(4*i-207)] <- X[,(i+3*ncol(Y3)+1)]
  }
  
  ###
  X.out2_1 <- X.out[1:44] %>% replace(!0, 0) 
  
  for(i in 1:11){
    X.out2_1[(4*i)] <- X.out[(i+1)]
    X.out2_1[(4*i-1)] <- X.out[(i+ncol(Y3)+1)]
    X.out2_1[(4*i-2)] <- X.out[(i+2*ncol(Y3)+1)]
    X.out2_1[(4*i-3)] <- X.out[(i+3*ncol(Y3)+1)]
  }
  
  X.out2_2 <- X.out[1:36] %>% replace(!0, 0) 
  
  for(i in 12:20){
    X.out2_2[(4*i-44)] <- X.out[(i+1)]
    X.out2_2[(4*i-45)] <- X.out[(i+ncol(Y3)+1)]
    X.out2_2[(4*i-46)] <- X.out[(i+2*ncol(Y3)+1)]
    X.out2_2[(4*i-47)] <- X.out[(i+3*ncol(Y3)+1)]
  }
  
  X.out2_3 <- X.out[1:124] %>% replace(!0, 0) 
  
  for(i in 21:51){
    X.out2_3[(4*i-80)] <- X.out[(i+1)]
    X.out2_3[(4*i-81)] <- X.out[(i+ncol(Y3)+1)]
    X.out2_3[(4*i-82)] <- X.out[(i+2*ncol(Y3)+1)]
    X.out2_3[(4*i-83)] <- X.out[(i+3*ncol(Y3)+1)]
  }
  
  X.out2_4 <- X.out[1:12] %>% replace(!0, 0) 
  
  for(i in 52:54){
    X.out2_4[(4*i-204)] <- X.out[(i+1)]
    X.out2_4[(4*i-205)] <- X.out[(i+ncol(Y3)+1)]
    X.out2_4[(4*i-206)] <- X.out[(i+2*ncol(Y3)+1)]
    X.out2_4[(4*i-207)] <- X.out[(i+3*ncol(Y3)+1)]
  }
  
  ###  
  X.arr1 = array(
    data = as.numeric(unlist(X2_1)),
    dim = c(nrow(X), 4, 11))
  
  X.arr2 = array(
    data = as.numeric(unlist(X2_2)),
    dim = c(nrow(X), 4, 9))
  
  X.arr3 = array(
    data = as.numeric(unlist(X2_3)),
    dim = c(nrow(X), 4, 31))
  
  X.arr4 = array(
    data = as.numeric(unlist(X2_4)),
    dim = c(nrow(X), 4, 3))
  
  ###
  X.out.arr1 = array(
    data = as.numeric(unlist(X.out2_1)),
    dim = c(1, 4, 11))
  
  X.out.arr2 = array(
    data = as.numeric(unlist(X.out2_2)),
    dim = c(1, 4, 9))
  
  X.out.arr3 = array(
    data = as.numeric(unlist(X.out2_3)),
    dim = c(1, 4, 31))
  
  X.out.arr4 = array(
    data = as.numeric(unlist(X.out2_4)),
    dim = c(1, 4, 3))
  
  batch_size = batch_size
  epochs = 100
  unit_n = unit_n
  
  input_layer_1 <- layer_input(shape = c(4, 31))
  input_layer_2 <- layer_input(shape = c(4, 9))
  input_layer_3 <- layer_input(shape = c(4, 11))
  input_layer_4 <- layer_input(shape = c(4, 3))
  
  # Level-1 LSTM layers
  lstm_layer_1_1 <- layer_lstm(units = unit_n, return_sequences = TRUE)(input_layer_1)
  lstm_layer_1_2 <- layer_lstm(units = unit_n, return_sequences = TRUE)(input_layer_2)
  
  # Concatenation after Level-1
  concatenated_layer_1 <- layer_concatenate(c(lstm_layer_1_1, lstm_layer_1_2))
  
  # Level-2 LSTM layer
  lstm_layer_3_1 <- layer_lstm(units = unit_n, return_sequences = TRUE)(input_layer_3)
  lstm_layer_2 <- layer_lstm(units = unit_n, return_sequences = TRUE)(concatenated_layer_1)
  
  # Concatenation for Level-2 and Level-3
  concatenated_layer_2 <- layer_concatenate(c(lstm_layer_2, lstm_layer_3_1))
  
  # Level-3 LSTM layer
  lstm_layer_3_2 <- layer_lstm(units = unit_n, return_sequences = TRUE)(input_layer_4)
  
  # Concatenation for Level-3
  concatenated_layer_3 <- layer_concatenate(c(concatenated_layer_2, lstm_layer_3_2))
  
  # Level-4 LSTM layer
  lstm_layer_4 <- layer_lstm(units = unit_n)(concatenated_layer_3)
  
  # Output layer
  output_layer <- layer_dense(units = 1)(lstm_layer_4)
  
  # Model
  model <- keras_model(inputs = list(input_layer_1, input_layer_2, input_layer_3, input_layer_4), 
                       outputs = output_layer)
  
  # Compile the model
  model %>% compile(loss = 'mse', optimizer = 'adam')
  
  # Fit the model to the training data
  model %>% summary()
  
  history = model %>% fit(x = list(X.arr3, X.arr2, X.arr1, X.arr4), y, epochs = epochs,
                          batch_size = batch_size, shuffle = FALSE, verbose = 2)
  
  pred = model(list(X.out.arr3, X.out.arr2, X.out.arr1, X.out.arr4)) %>%
    denormalize(min(Y2[,indice]),max(Y2[,indice]))
  
  return(list("model"=model,"pred"=pred))
}

###### Multi-Input LSTM ######
run_multiinput_lstm_o=function(Y,indice,lag, batch_size = 30, unit_n = 32){
  horizon = lag
  Y2 = Y %>% as.data.frame()
  Y3 = lapply(Y2, normalize) %>% as.data.frame() %>% as.matrix()
  
  # aux=embed(Y3,4+lag)
  # y=aux[,indice]
  # X=aux[,-c(1:(ncol(Y3)*lag))]
  # 
  # if(lag==1){
  #   X.out=tail(aux,1)[1:ncol(X)]  
  # }else{
  #   X.out=aux[,-c(1:(ncol(Y3)*(lag-1)))]
  #   X.out=tail(X.out,1)[1:ncol(X)]
  # }
  
  X=embed(as.matrix(Y3),4)
  
  Xin=X[-c((nrow(X)-horizon+1):nrow(X)),]
  Xout=X[nrow(X),]
  
  y=tail(Y3[,1],nrow(Xin))
  X = Xin 
  X.out = Xout
  
  ###
  X2_1 <- X[,1:44] %>% replace(!0, 0) 
  
  for(i in 1:11){
    X2_1[,(4*i)] <- X[,(i+1)]
    X2_1[,(4*i-1)] <- X[,(i+ncol(Y3)+1)]
    X2_1[,(4*i-2)] <- X[,(i+2*ncol(Y3)+1)]
    X2_1[,(4*i-3)] <- X[,(i+3*ncol(Y3)+1)]
  }
  
  X2_2 <- X[,1:36] %>% replace(!0, 0) 
  
  for(i in 12:20){
    X2_2[,(4*i-44)] <- X[,(i+1)]
    X2_2[,(4*i-45)] <- X[,(i+ncol(Y3)+1)]
    X2_2[,(4*i-46)] <- X[,(i+2*ncol(Y3)+1)]
    X2_2[,(4*i-47)] <- X[,(i+3*ncol(Y3)+1)]
  }
  
  X2_3 <- X[,1:124] %>% replace(!0, 0) 
  
  for(i in 21:51){
    X2_3[,(4*i-80)] <- X[,(i+1)]
    X2_3[,(4*i-81)] <- X[,(i+ncol(Y3)+1)]
    X2_3[,(4*i-82)] <- X[,(i+2*ncol(Y3)+1)]
    X2_3[,(4*i-83)] <- X[,(i+3*ncol(Y3)+1)]
  }
  
  X2_4 <- X[,1:12] %>% replace(!0, 0) 
  
  for(i in 52:54){
    X2_4[,(4*i-204)] <- X[,(i+1)]
    X2_4[,(4*i-205)] <- X[,(i+ncol(Y3)+1)]
    X2_4[,(4*i-206)] <- X[,(i+2*ncol(Y3)+1)]
    X2_4[,(4*i-207)] <- X[,(i+3*ncol(Y3)+1)]
  }
  
  ###
  X.out2_1 <- X.out[1:44] %>% replace(!0, 0) 
  
  for(i in 1:11){
    X.out2_1[(4*i)] <- X.out[(i+1)]
    X.out2_1[(4*i-1)] <- X.out[(i+ncol(Y3)+1)]
    X.out2_1[(4*i-2)] <- X.out[(i+2*ncol(Y3)+1)]
    X.out2_1[(4*i-3)] <- X.out[(i+3*ncol(Y3)+1)]
  }
  
  X.out2_2 <- X.out[1:36] %>% replace(!0, 0) 
  
  for(i in 12:20){
    X.out2_2[(4*i-44)] <- X.out[(i+1)]
    X.out2_2[(4*i-45)] <- X.out[(i+ncol(Y3)+1)]
    X.out2_2[(4*i-46)] <- X.out[(i+2*ncol(Y3)+1)]
    X.out2_2[(4*i-47)] <- X.out[(i+3*ncol(Y3)+1)]
  }
  
  X.out2_3 <- X.out[1:124] %>% replace(!0, 0) 
  
  for(i in 21:51){
    X.out2_3[(4*i-80)] <- X.out[(i+1)]
    X.out2_3[(4*i-81)] <- X.out[(i+ncol(Y3)+1)]
    X.out2_3[(4*i-82)] <- X.out[(i+2*ncol(Y3)+1)]
    X.out2_3[(4*i-83)] <- X.out[(i+3*ncol(Y3)+1)]
  }
  
  X.out2_4 <- X.out[1:12] %>% replace(!0, 0) 
  
  for(i in 52:54){
    X.out2_4[(4*i-204)] <- X.out[(i+1)]
    X.out2_4[(4*i-205)] <- X.out[(i+ncol(Y3)+1)]
    X.out2_4[(4*i-206)] <- X.out[(i+2*ncol(Y3)+1)]
    X.out2_4[(4*i-207)] <- X.out[(i+3*ncol(Y3)+1)]
  }
  
  ###  
  X.arr1 = array(
    data = as.numeric(unlist(X2_1)),
    dim = c(nrow(X), 4, 11))
  
  X.arr2 = array(
    data = as.numeric(unlist(X2_2)),
    dim = c(nrow(X), 4, 9))
  
  X.arr3 = array(
    data = as.numeric(unlist(X2_3)),
    dim = c(nrow(X), 4, 31))
  
  X.arr4 = array(
    data = as.numeric(unlist(X2_4)),
    dim = c(nrow(X), 4, 3))
  
  ###
  X.out.arr1 = array(
    data = as.numeric(unlist(X.out2_1)),
    dim = c(1, 4, 11))
  
  X.out.arr2 = array(
    data = as.numeric(unlist(X.out2_2)),
    dim = c(1, 4, 9))
  
  X.out.arr3 = array(
    data = as.numeric(unlist(X.out2_3)),
    dim = c(1, 4, 31))
  
  X.out.arr4 = array(
    data = as.numeric(unlist(X.out2_4)),
    dim = c(1, 4, 3))
  
  batch_size = batch_size
  epochs = 100
  unit_n = unit_n
  
  input_layer_1 <- layer_input(shape = c(4, 31))
  input_layer_2 <- layer_input(shape = c(4, 9))
  input_layer_3 <- layer_input(shape = c(4, 11))
  input_layer_4 <- layer_input(shape = c(4, 3))
  
  # Input LSTM layers
  lstm_layer_1_1 <- layer_lstm(units = unit_n, return_sequences = TRUE)(input_layer_1)
  lstm_layer_1_2 <- layer_lstm(units = unit_n, return_sequences = TRUE)(input_layer_2)
  lstm_layer_1_3 <- layer_lstm(units = unit_n, return_sequences = TRUE)(input_layer_3)
  lstm_layer_1_4 <- layer_lstm(units = unit_n, return_sequences = TRUE)(input_layer_4)
  
  # Concatenation
  concatenated_layer <- layer_concatenate(c(lstm_layer_1_1, lstm_layer_1_2, lstm_layer_1_3, lstm_layer_1_4))
  lstm_layer_2 <- layer_lstm(units = unit_n)(concatenated_layer)
  
  # Output layer
  output_layer <- layer_dense(units = 1)(lstm_layer_2)
  
  # Model
  model <- keras_model(inputs = list(input_layer_1, input_layer_2, input_layer_3, input_layer_4), 
                       outputs = output_layer)
  
  # Compile the model
  model %>% compile(loss = 'mse', optimizer = 'adam')
  
  # Fit the model to the training data
  model %>% summary()
  
  history = model %>% fit(x = list(X.arr3, X.arr2, X.arr1, X.arr4), y, epochs = epochs,
                          batch_size = batch_size, shuffle = FALSE, verbose = 2)
  
  pred = model(list(X.out.arr3, X.out.arr2, X.out.arr1, X.out.arr4)) %>%
    denormalize(min(Y2[,indice]),max(Y2[,indice]))
  
  return(list("model"=model,"pred"=pred))
}

###### Rolling Window function ######
rolling.window.lstm.single=function(Y, npred, indice=1, lag=1, batch = 30, unit = 32){
  horizon = lag
  save.pred=matrix(NA,npred-horizon+1,1)
  
  for(i in npred:horizon){
    Y.window=Y[(1+npred-i):(nrow(Y)-i),] %>% as.data.frame()
    lstm=run_single_lstm(Y.window,indice,lag,batch,unit)
    save.pred[(1+npred-i),]=as.numeric(lstm$pred) # Note as.numeric()
    cat("iteration",(1+npred-i),"\n")
  }
  
  real=Y[,indice]
  plot(real,type="l")
  lines(c(rep(NA,length(real)-npred+horizon-1),save.pred),col="red")
  
  rmse=sqrt(mean((tail(real,npred-horizon+1)-save.pred)^2))
  mae=mean(abs(tail(real,npred-horizon+1)-save.pred))
  errors=c("rmse"=rmse,"mae"=mae)
  
  return(list("pred"=save.pred,"errors"=errors))
}

rolling.window.lstm=function(Y,npred,indice=1,lag=1, batch = 30, unit = 32){
  
  horizon = lag
  save.pred=matrix(NA,npred-horizon+1,1)
  
  for(i in npred:horizon){
    Y.window=Y[(1+npred-i):(nrow(Y)-i),] %>% as.data.frame()
    lstm=run_multi_lstm(Y.window,indice,lag,batch,unit)
    save.pred[(1+npred-i),]=as.numeric(lstm$pred) # Note as.numeric()
    cat("iteration",(1+npred-i),"\n")
  }
  
  real=Y[,indice]
  plot(real,type="l")
  lines(c(rep(NA,length(real)-npred+horizon-1),save.pred),col="red")
  
  rmse=sqrt(mean((tail(real,npred-horizon+1)-save.pred)^2))
  mae=mean(abs(tail(real,npred-horizon+1)-save.pred))
  errors=c("rmse"=rmse,"mae"=mae)
  
  return(list("pred"=save.pred,"errors"=errors))
}

rolling.window.lstm.minput=function(Y, npred, indice=1, lag=1, batch = 30, unit = 32){
  
  horizon = lag
  save.pred=matrix(NA,npred-horizon+1,1)
  
  for(i in npred:horizon){
    Y.window=Y[(1+npred-i):(nrow(Y)-i),] %>% as.data.frame()
    lstm=run_multiinput_lstm(Y.window,indice,lag,batch,unit)
    save.pred[(1+npred-i),]=as.numeric(lstm$pred) # Note as.numeric()
    cat("iteration",(1+npred-i),"\n")
  }
  
  real=Y[,indice]
  plot(real,type="l")
  lines(c(rep(NA,length(real)-npred+horizon-1),save.pred),col="red")
  
  rmse=sqrt(mean((tail(real,npred-horizon+1)-save.pred)^2))
  mae=mean(abs(tail(real,npred-horizon+1)-save.pred))
  errors=c("rmse"=rmse,"mae"=mae)
  
  return(list("pred"=save.pred,"errors"=errors))
}

rolling.window.lstm.minput.o=function(Y, npred, indice=1, lag=1, batch = 30, unit = 32){
  
  horizon = lag
  save.pred=matrix(NA,npred-horizon+1,1)
  
  for(i in npred:horizon){
    Y.window=Y[(1+npred-i):(nrow(Y)-i),] %>% as.data.frame()
    lstm=run_multiinput_lstm_o(Y.window,indice,lag,batch,unit)
    save.pred[(1+npred-i),]=as.numeric(lstm$pred) # Note as.numeric()
    cat("iteration",(1+npred-i),"\n")
  }
  
  real=Y[,indice]
  plot(real,type="l")
  lines(c(rep(NA,length(real)-npred+horizon-1),save.pred),col="red")
  
  rmse=sqrt(mean((tail(real,npred-horizon+1)-save.pred)^2))
  mae=mean(abs(tail(real,npred-horizon+1)-save.pred))
  errors=c("rmse"=rmse,"mae"=mae)
  
  return(list("pred"=save.pred,"errors"=errors))
}