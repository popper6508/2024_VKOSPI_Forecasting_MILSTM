runnn=function(Y,indice,horizon){
  Y2=cbind(Y)
  X=embed(as.matrix(Y2),4)
  
  Xin=X[-c((nrow(X)-horizon+1):nrow(X)),]
  Xout=X[nrow(X),]
  
  y=tail(Y2[,1],nrow(Xin))
  X = Xin 
  X.out = Xout
  ##
  trainingframe = as.h2o(cbind(y=y,X))
  
  model = h2o.deeplearning(y = 'y',
                           training_frame = trainingframe,
                           activation = 'Rectifier',
                           hidden = c(32,16,8),
                           epochs = 100,
                           train_samples_per_iteration = -2,
                           seed = 1)
  
  xoutframe = t(c(NA,X.out))
  colnames(xoutframe) = colnames(trainingframe)
  xoutframe = as.h2o(xoutframe)
  
  y_pred = h2o.predict(model, newdata = xoutframe)
  pred = as.vector(y_pred)
  
  return(list("model"=model,"pred"=pred))
}


nn.rolling.window=function(Y,npred,indice=1,horizon=1){
  
  save.pred=matrix(NA,npred-horizon+1,1)
  for(i in npred:horizon){
    Y.window=Y[(1+npred-i):(nrow(Y)-i),]
    lasso=runnn(Y.window,indice,horizon)
    save.pred[(1+npred-i),]=lasso$pred
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

