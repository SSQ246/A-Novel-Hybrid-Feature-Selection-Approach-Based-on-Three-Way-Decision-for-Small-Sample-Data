#Read the dataset
library(caret)
mfeat_zer <- read.table('C:\\Users\\QAQ\\Desktop\\r\\0-9分类\\mfeat-zer')
mfeat_pix <- read.table('C:\\Users\\QAQ\\Desktop\\r\\0-9分类\\mfeat-pix')
mfeat_mor <- read.table('C:\\Users\\QAQ\\Desktop\\r\\0-9分类\\mfeat-mor')
mfeat_kar <- read.table('C:\\Users\\QAQ\\Desktop\\r\\0-9分类\\mfeat-kar')
mfeat_fou <- read.table('C:\\Users\\QAQ\\Desktop\\r\\0-9分类\\mfeat-fou')
mfeat_fac <- read.table('C:\\Users\\QAQ\\Desktop\\r\\0-9分类\\mfeat-fac')
r <- rep(0:9,each=200)

set.seed(1)
library(e1071)
library(randomForest)
library(Boruta)
#defined function svm
svm_fun <- function(a,b,c,d,e){
  x_fun <- e[,a]
  svm_fun <- svm(x_fun, b)
  cx_fun <- c[,a]
  p <<- predict(svm_fun,newdata=cx_fun)
  p <<- round(p)
  a=0
  for(i in 1:nrow(c)){
    if(p[i]==d[i]){
      a=a+1
    }
  }
  accuracy <- a/nrow(c)
  return(accuracy)
}
#cross validation
cv_re <- function(fea,td){
  p <- 0
  k <- 5
  to_va <- matrix(nrow = 0, ncol = 650)
  for(i in k:1){
    index <- caret::createDataPartition(td$r, p = 1/i, list = FALSE, times = 1)
    va <- td[index,]
    td <- td[-index,]
    train <- rbind(to_va,td)
    to_va <- rbind(to_va,va)
    x <- train[,2:650]
    y <- train[,1]
    x_t <- va[,2:650]
    y_t <- va[,1]
    p <- p + svm_fun(fea,y,x_t,y_t,x)
  }
  return(p/k)
}
#Define the function 'best' to select a subset of features
best <- function(sorted_fea,td){
  aa <- 1
  p <- 0
  for(i in 1:50){
    fea_l <- sorted_fea[1:i]
    t <- cv_re(fea_l,td)
    if(t>p){
      p <- t
      aa <- length(fea_l)
    }
  }
  pp <- c(aa,p)
  return(pp)
}
#Define function f1 and calculate its value
f1 <- function(fea,result){
  f1_p <- 0
  f1_r <- 0
  for(i in 0:9){
    f1_a <- which(fea == i)
    f1_b <- which(result == i)
    f1_c <- which(f1_a %in% f1_b)
    f1_p <- f1_p + length(f1_c)/length(f1_b)
    f1_r <- f1_r + length(f1_c)/length(f1_a)
  }
  f1_p <- f1_p/10
  f1_r <- f1_r/10
  return((2*f1_p*f1_r)/(f1_p + f1_r))
}
dat <- cbind(r,mfeat_zer,mfeat_pix,mfeat_mor,mfeat_kar,mfeat_fou,mfeat_fac)
colnames(dat) <- c("r",paste0("x",1:(ncol(dat)-1)))
#Modify the size of the training set here
testIndex <- caret::createDataPartition(dat$r, p = 0.9, list = FALSE, times = 1)
testData <- dat[testIndex, ]
x_test <- testData[, -1]
y_test <- testData[, 1]
Data <- dat[-testIndex, ]
x <- Data[,2:650]
y <- Data[,1]


#lasso
stime <- Sys.time()
library(glmnet)
x <- as.matrix(x)
x_lasso <- scale(x)
fit <- cv.glmnet(x_lasso, y , alpha=1)
f <- coef(fit)
f <- f[-1]
f <- abs(f)
ma_lasso <- cbind(c(1:length(f)),f)
ma_lasso <- ma_lasso[order(ma_lasso[, 2], decreasing = TRUE), ]
fea_lasso <- ma_lasso[,1]
acc_lasso <- best(fea_lasso,Data)
acc_lasso <- c(acc_lasso,svm_fun(fea_lasso[1:acc_lasso[1]],y,x_test,y_test,x),f1(y_test,p))
etime <- Sys.time()
etime - stime



#correlation coefficient
stime <- Sys.time()
library(mlr)
myFUN <- function(x){cor(x,Data[,1])}
result <- apply(Data[,2:ncol(Data)],2,myFUN)
result <- abs(result)
aaa <- cbind(c(1:length(result)),result)
result <- result/max(result)
fea_re <- order(result,decreasing = TRUE)
acc_re <- best(fea_re,Data)
acc_re <- c(acc_re,svm_fun(fea_re[1:acc_re[1]],y,x_test,y_test,x),f1(y_test,p))
etime <- Sys.time()
etime - stime






#Impurity
stime <- Sys.time()
library(mlr3verse)
library(dplyr)
library(ranger)
task <- as_task_classif(Data,target = "r")
lrn = lrn("classif.ranger", importance = "impurity")
flt <- flt("importance",learner=lrn)
flt$calculate(task)
flt <- as.data.table(flt)
fea_im <- c()
for(i in 1:nrow(flt)){
  fea_im <- c(fea_im,which(names(Data)==as.character(flt[i,1])))
}
fea_im <- fea_im-1
acc_im <- best(fea_im,Data)
acc_im <- c(acc_im,svm_fun(fea_im[1:acc_im[1]],y,x_test,y_test,x),f1(y_test,p))
etime <- Sys.time()
etime - stime




#Mutual Information
stime <- Sys.time()
library(infotheo)
x_inf <- discretize(Data[,2:650],"equalfreq",10)
inf <- c()
for(i in 1:ncol(x)){
  inf <- c(inf,mutinformation(y, x_inf[,i]))
}
fea_inf <- order(inf,decreasing = TRUE)
acc_inf <- best(fea_inf,Data)
acc_inf <- c(acc_inf,svm_fun(fea_inf[1:acc_inf[1]],y,x_test,y_test,x),f1(y_test,p))
etime <- Sys.time()
etime - stime


#Boruta
stime <- Sys.time()
boruta <- Boruta(r~., data=Data)
bo_li <- attStats(boruta)
bo_1 <- bo_li[,1]
fea_bo <- order(bo_1,decreasing = TRUE)
acc_bo <- best(fea_bo,Data)
acc_bo <- c(acc_bo,svm_fun(fea_inf[1:acc_bo[1]],y,x_test,y_test,x),f1(y_test,p))
etime <- Sys.time()
etime - stime


#Random Forest
stime <- Sys.time()
rf <- randomForest(r~., data=Data, importance=TRUE, proximity=TRUE,type="classification") 
ran <- round(randomForest::importance(rf), 2)
ran_1 <- ran[,-1]
ran_2 <- ran[,-2]
fea_ran <- order(ran_1,decreasing = TRUE)
acc_ran <- best(fea_ran,Data)
acc_ran <- c(acc_ran,svm_fun(fea_ran[1:acc_ran[1]],y,x_test,y_test,x),f1(y_test,p))
fea_ran_2 <- order(ran_2,decreasing = TRUE)
acc_ran_2 <- best(fea_ran_2,Data)
acc_ran_2 <- c(acc_ran_2,svm_fun(fea_ran_2[1:acc_ran_2[1]],y,x_test,y_test,x),f1(y_test,p))
etime <- Sys.time()
etime - stime









#H-TwD
#Only the methods that need to be mixed are retained here
stime <- Sys.time()
fea_1 <- c()#fea_re[1:acc_re[1]]  
fea_2 <- fea_ran[1:acc_ran[1]] 
fea_3 <- c()#fea_ran_2[1:acc_ran_2[1]]
fea_4 <- fea_im[1:acc_im[1]] 
fea_5 <- c()#fea_lasso[1:acc_lasso[1]]
fea_6 <- fea_bo[1:acc_bo[1]]
fea_7 <- c()#fea_inf[1:acc_inf[1]]
fea_he <- unique(c(fea_1,fea_2,fea_3,fea_4,fea_5,fea_6,fea_7))
hebing_ma <- cbind(seq(1,ncol(x)),rep(0,times=ncol(x)))
for(i in 1:nrow(hebing_ma)){
  if(hebing_ma[i,1] %in% fea_1){
    hebing_ma[i,2] <- hebing_ma[i,2] + acc_re[2] - 0.001*which(fea_1 == hebing_ma[i,1])
  }
  if(hebing_ma[i,1] %in% fea_2){
    hebing_ma[i,2] <- hebing_ma[i,2] + acc_ran[2] - 0.001*which(fea_2 == hebing_ma[i,1])
  }
  if(hebing_ma[i,1] %in% fea_3){
    hebing_ma[i,2] <- hebing_ma[i,2] + acc_ran_2[2] - 0.001*which(fea_3 == hebing_ma[i,1])
  }
  if(hebing_ma[i,1] %in% fea_4){
    hebing_ma[i,2] <- hebing_ma[i,2] + acc_im[2] - 0.001*which(fea_4 == hebing_ma[i,1])
  }
  if(hebing_ma[i,1] %in% fea_5){
    hebing_ma[i,2] <- hebing_ma[i,2] + acc_lasso[2] - 0.001*which(fea_5 == hebing_ma[i,1])
  }
  if(hebing_ma[i,1] %in% fea_6){
    hebing_ma[i,2] <- hebing_ma[i,2] + acc_bo[2] - 0.001*which(fea_6 == hebing_ma[i,1])
  }
  if(hebing_ma[i,1] %in% fea_7){
    hebing_ma[i,2] <- hebing_ma[i,2] + acc_inf[2] - 0.001*which(fea_7 == hebing_ma[i,1])
  }
}
hebing_ma <- hebing_ma[order(hebing_ma[, 2], decreasing = TRUE), ]
a_i <- 0
a_j <- 0
p <- 0
fea <- hebing_ma[1:length(fea_he),1]
#Set feature weight threshold here
count <- length(which(hebing_ma[,2] > 1.5))
num_he <- length(which(hebing_ma[,2] > 0.8))
num_dan <- max(max(which(hebing_ma[,1] %in% fea_6)),num_he)
count
num_he
num_dan
length(fea)
etime <- Sys.time()
etime - stime

#H-TwD
stime <- Sys.time()
p_re <- 0
i <- count
fea_xuan = fea[1:i]
fea_bian = fea[(i+1):num_dan]
pp = cv_re(fea_xuan,Data)
for(j in fea_bian){
  arr_mid = c(fea_xuan,j)
  p_mid = cv_re(arr_mid,Data)
  if(p_mid > pp){
    fea_xuan = arr_mid
    pp = p_mid
    a_jj = j
  }
}
if(pp > p_re){
  arr_re = fea_xuan
  p_re = pp
  a_ii = i
}
a_ii
a_jj
arr_re
print("无")
p_re
svm_fun(arr_re,y,x_test,y_test,x)
f1(p,y_test)
etime <- Sys.time()
etime - stime
