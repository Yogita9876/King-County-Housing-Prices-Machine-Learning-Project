house <- read.csv("kc_house_dataRowremoved.csv", header = TRUE)
NROW(house)

price = house$price

house$price = NULL
house["price"] <- price

    house$bedrooms.f=factor(house$bedrooms)
    house$bathrooms.f=factor(house$bathrooms)
    house$floors.f=factor(house$floors)
    house$waterfront.f=factor(house$waterfront)
    house$view.f=factor(house$view)
    house$condition.f=factor(house$condition)
    house$grade.f=factor(house$grade)
    house$yr_built.f=factor(house$yr_built)
    house$yr_renovated.f=factor(house$yr_renovated)
    house$zipcode.f=factor(house$zipcode)
    
    bedrooms = model.matrix(~house$bedrooms.f)
    bathrooms = model.matrix(~house$bathrooms.f)
    floors = model.matrix(~house$floors.f)
    waterfront = model.matrix(~house$waterfront.f)
    view = model.matrix(~house$view.f)
    condition = model.matrix(~house$condition.f)
    grade = model.matrix(~house$grade.f)
    yr_built = model.matrix(~house$yr_built.f)
    yr_renovated = model.matrix(~house$yr_renovated.f)
    zipcode = model.matrix(~house$zipcode.f)

DF=house[,c(3,4,10,16,17)];
house$

normalizedData=apply(DF,2, function(x)((x-mean(x))/sd(x)));
data = data.frame(normalizedData)

data=cbind(data,data.frame(bedrooms[,-1]),data.frame(bathrooms[,-1]),data.frame(floors[,-1]),data.frame(waterfront[,-1]),
           data.frame(view[,-1]),data.frame(condition[,-1]),data.frame(grade[,-1]),data.frame(yr_built[,-1]),data.frame(yr_renovated[,-1]),data.frame(zipcode[,-1]),price);

NROW(na.omit(data))

ncol(data)

write.csv(data,file='new_house_data_set_4.csv',row.names=FALSE)

ML_Linear = lm(price~., data = data)

summary(ML_Linear)

plot(ML_Linear)

data1 = subset(data,data$price<3000000)
data1 = subset(data1,data$price>100000)

summary(data1)

write.csv(data1,file='new_house_data_set_5.csv',row.names=FALSE)

NROW(na.omit(data1))

summary(data1)
NROW((data1))
NROW((data))

house$


library(glmnet)

x=model.matrix(data[,321]~as.matrix(data[,1:320]))
y=data[,321]

cv.fit=cv.glmnet(x,y,alpha=1,nfolds = 10)

varImp(ML_Linear)

plot(cv.fit)
plot(cv.fit$glmnet.fit,"lambda",lable= TRUE);

plot(cv.fit$glmnet.fit,"lambda",label = TRUE);

cv.fit$lambda.min
cv.fit$lambda.1se
cv.fit$glmnet.fit

Cofficients(cv.fit$lambda.min)


myModel = glmnet(x,y,alpha=1,lambda = 51570.00)



yhat=predict(myModel,x)
MSE=sqrt(sum((yhat-y)^2)/nrow(x))
rsquare=1-(sum((yhat-y)^2)/sum((y-mean(y))^2))
adjrsquare=1-(((1-rsquare)*(nrow(x)-1))/(nrow(x)-sum(myModel$beta != 0)-1))


coefficients(myModel)

house.zipcode.f98004
data$house.

summary()

house2 <- read.csv("/Users/baderalbulayhis/Desktop/new_house_data_set_5.csv", header = TRUE)

summary(house2)
NROW(na.omit(house2))


ncol(house2)

x=model.matrix(house2[,322]~as.matrix(house2[,1:321]))
y=house2[,322]

ML_Linear = lm(house2$price~., data = data1)

plot(ML_Linear)

summary(ML_Linear)

cv.fit=cv.glmnet(x,y,alpha=1,nfolds = 10)

plot(cv.fit)
plot(cv.fit$glmnet.fit,"lambda",label = TRUE)

myModel = glmnet(x,y,alpha=1,lambda = 4721)



