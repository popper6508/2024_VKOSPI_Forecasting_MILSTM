runAR=function(Y,indice,lag,what,type="fixed"){
  X = Y %>% na.omit()
  
  Xin=X[-c((nrow(X)-lag+1):nrow(X)),]
  Xout=X[nrow(X),]
  
  y=tail(Y[,1],nrow(Xin))
  X = Xin 
  X.out = Xout
  
  if(type=="fixed"){
    model=lm(y~X)
    coef=coef(model)
  }
  
  if(type=="bic"){
    bb=Inf
    for(i in seq(1,ncol(X),1)){
      m=lm(y~X[,1:i])
      crit=BIC(m)
      if(crit<bb){
        bb=crit
        model=m
        ar.coef=coef(model)
      }
    }
    coef=rep(0,ncol(X)+1)
    coef[1:length(ar.coef)]=ar.coef
  }
  pred=c(1,X.out)%*%coef
  
  return(list("model"=model,"pred"=pred,"coef"=coef))
}

runAR2=function(Y,indice,lag,what,type="fixed"){
  horizon = lag
  X = Y %>% na.omit()
  
  Xin=X[-c((nrow(X)-horizon+1):nrow(X)),(indice+1):ncol(X)]
  Xout=X[nrow(X),(indice+1):ncol(X)]
  y = X[-c((nrow(X)-horizon+1):nrow(X)),indice]

  model=ic.glmnet(Xin,y,alpha = 1)
  lasso_selected = names(model$coef != 0)[(indice+1):length((model$coef != 0))]

  X = Xin[,lasso_selected]
  Xout = Xout[lasso_selected]
  
  if(type=="fixed"){
    model=lm(y~X)
    coef=coef(model)
  }
  
  if(type=="bic"){
    bb=Inf
    for(i in seq(1,ncol(X),1)){
      m=lm(y~X[,1:i])
      crit=BIC(m)
      if(crit<bb){
        bb=crit
        model=m
        ar.coef=coef(model)
      }
    }
    coef=rep(0,ncol(X)+1)
    coef[1:length(ar.coef)]=ar.coef
  }
  pred=c(1,Xout)%*%coef
  
  return(list("model"=model,"pred"=pred))
}

ar.rolling.window=function(Y,npred,indice=1,lag=1,type="fixed", what = "harx"){
  
  horizon = lag
  
  save.coef=matrix(NA,npred,18)
  save.pred=matrix(NA,npred-horizon+1,1)
  
  for(i in npred:horizon){
    Y.window=Y[(1+npred-i):(nrow(Y)-i),]
    fact=runAR(Y.window,indice,lag, what)
    
    save.pred[(1+npred-i),]=fact$pred
    cat("iteration",(1+npred-i),"\n")
  }
  
  real=Y[,indice]
  plot(real,type="l")
  lines(c(rep(NA,length(real)-npred+horizon-1),save.pred),col="red")
  
  rmse=sqrt(mean((tail(real,npred-horizon+1)-save.pred)^2))
  mae=mean(abs(tail(real,npred-horizon+1)-save.pred))
  errors=c("rmse"=rmse,"mae"=mae)
  
  return(list("pred"=save.pred,"coef"=save.coef,"errors"=errors))
}

ar.rolling.window2=function(Y,npred,indice=1,lag=1,type="fixed", what = "harx"){
  
  horizon = lag

  save.pred=matrix(NA,npred-horizon+1,1)
  
  for(i in npred:horizon){
    Y.window=Y[(1+npred-i):(nrow(Y)-i),]
    fact=runAR2(Y.window,indice,lag, what)
    
    save.pred[(1+npred-i),]=fact$pred
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