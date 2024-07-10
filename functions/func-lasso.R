runlasso=function(Y,indice,horizon,alpha=1,type="lasso"){
  Y2=Y
  X=embed(as.matrix(Y2),4)
  
  Xin=X[-c((nrow(X)-horizon+1):nrow(X)),]
  Xout=X[nrow(X),]
  
  y=tail(Y2[,1],nrow(Xin))
  X = Xin 
  X.out = Xout
  ##
  
  model=ic.glmnet(X,y,alpha = alpha)
  coef=model$coef
  if(type=="adalasso"){
    penalty=(abs(coef[-1])+1/sqrt(length(y)))^(-1)
    model=ic.glmnet(X,y,penalty.factor = penalty,alpha=alpha)
  }
  
  if(type=="fal"){
    taus=c(seq(0.1,1,0.1),1.25,1.5,2,3,4,5,7,10)
    alphas=seq(0,1,0.1)
    bb=Inf
    for(alpha in alphas){
      m0=ic.glmnet(X,y,alpha = alpha)
      coef=m0$coef
      for(tau in taus){
        penalty=(abs(coef[-1])+1/sqrt(length(y)))^(-tau)
        m=ic.glmnet(X,y,penalty.factor = penalty)
        crit=m$bic
        if(crit<bb){
          model=m
          bb=crit
        }
      }
    }
  }
  pred=predict(model,X.out)
  
  return(list("model"=model,"pred"=pred))
}

lasso.rolling.window=function(Y,npred,indice=1,horizon=1,alpha=1,type="lasso"){
  
  save.model=list()
  save.pred=matrix(NA,npred-horizon+1,1)
  for(i in npred:horizon){
    Y.window=Y[(1+npred-i):(nrow(Y)-i),]
    lasso=runlasso(Y.window,indice,horizon,alpha,type)
    save.model[(1+npred-i)]=lasso$model
    save.pred[(1+npred-i),]=lasso$pred
    cat("iteration",(1+npred-i),"\n")
  }
  
  real=Y[,indice]
  plot(real,type="l")
  lines(c(rep(NA,length(real)-npred+horizon-1),save.pred),col="red")
  
  rmse=sqrt(mean((tail(real,npred-horizon+1)-save.pred)^2))
  mae=mean(abs(tail(real,npred-horizon+1)-save.pred))
  errors=c("rmse"=rmse,"mae"=mae)
  
  return(list("pred"=save.pred,"coef"=save.model,"errors"=errors))
}
