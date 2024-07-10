runxgb=function(Y,indice,horizon){
  Y2=cbind(Y)
  X=embed(as.matrix(Y2),4)
  
  Xin=X[-c((nrow(X)-horizon+1):nrow(X)),]
  Xout=X[nrow(X),]
  
  y=tail(Y2[,1],nrow(Xin))
  X = Xin 
  X.out = Xout
  ##
  
  model = xgboost(X,label = y,nrounds = 1000, verbose = FALSE,
                  params=list(eta=0.05,nthread=1,colsample_bylevel=2/3,subsample=1,max_depth=4,min_child_weight=nrow(X)/200))
  
  pred=predict(model,t(X.out))
  
  return(list("model"=model,"pred"=pred))
}


xgb.rolling.window=function(Y,npred,indice=1,horizon=1){
  
  save.pred=matrix(NA,npred-horizon+1,1)
  for(i in npred:horizon){
    Y.window=Y[(1+npred-i):(nrow(Y)-i),]
    lasso=runxgb(Y.window,indice,horizon)
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