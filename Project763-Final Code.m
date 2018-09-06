%===============================================
%--KING COUNTY HOUSING ANALYSIS MATLAB CODE-----
%--SEIS 763 - MACHINE & DEEP LEARNING PROJECT---
%--07/17/2017   
%===============================================

%===============================================
%--DATA SET-UP & LINEAR MODEL ML1 SECTION------
%===============================================


%loading the Data
houseData = readtable('C:\temp\kc_house_data_BAD_DATA_REMOVED.xlsx');


%create dummy variables for bedroom, bathroom, 
%grades, floors, condition, yr_built, yr_renovated, zipcode

DVBedroom = dummyvar(nominal(houseData.bedrooms(:,end)));
DVBedroom = DVBedroom (:,2:end);

DVBathroom = dummyvar(nominal(houseData.bathrooms(:,end)));
DVBathroom = DVBathroom (:,2:end);

DVGrade = dummyvar(nominal(houseData.grade(:,end)));
DVGrade = DVGrade (:, 2:end);

DVFloors = dummyvar(nominal(houseData.floors(:,end)));
DVFloors = DVFloors (:, 2:end);

DVCondition = dummyvar(nominal(houseData.condition(:,end)));
DVCondition = DVCondition (:, 2:end);

DVYrbuilt = dummyvar(nominal(houseData.yr_built(:,end)));
DVYrbuilt = DVYrbuilt (:, 2:end);

DVYrrenovated = dummyvar(nominal(houseData.yr_renovated(:,end)));
DVYrrenovated = DVYrrenovated (:,2:end);

DVZipcode = dummyvar(nominal(houseData.zipcode(:,end)));
DVZipcode = DVZipcode(:, 2:end);

DVView = dummyvar(nominal(houseData.view(:,end)));
DVView = DVView (:,2:end);

DVWaterfront = dummyvar(nominal(houseData.waterfront(:,end)));
DVWaterfront = DVWaterfront (:,2:end);

%(zscore(houseData.sqft_basement))
%Creating the model
Xcat = [DVBedroom, DVBathroom, DVFloors, DVView , DVCondition,DVGrade, DVYrbuilt, DVYrrenovated, DVZipcode, DVWaterfront];
Xvar = zscore([houseData.sqft_living houseData.sqft_basement houseData.lat houseData.long houseData.sqft_lot]);
Xfinal = [Xvar Xcat];
Y = houseData.price;
md1 = fitlm (Xfinal, Y)
plot (md1)



%[b fitinfo] = lasso(X, Y, 'CV',10, 'Alpha', 1); 
%lassoPlot(b,fitinfo,'PlotType', 'Lambda', 'XScale', 'log')

%===============================================
%--FIND OUTLIERS & LINEAR MODEL ML2 SECTION-----
%===============================================

%find the outliers
potential_outlier_cooks=find((md1.Diagnostics.CooksDistance)>5*mean(md1.Diagnostics.CooksDistance));
potential_outlier=find((md1.Diagnostics.CooksDistance+md1.Diagnostics.Leverage)>5*mean(md1.Diagnostics.CooksDistance+md1.Diagnostics.Leverage));

plotDiagnostics(md1,'cookd')

%Remove the outliers from X
[filenew, t] = removerows (houseData, 'ind', potential_outlier(:));

%Remove the outliers from Y
%[Ynew, t1] = removerows(Y, 'ind', potential_outlier(:));

%Create new dummy variables with outliers remmoved

%based on removal of outlier for bedroom dummyvar
DVBedroomNew = dummyvar(nominal(filenew.bedrooms(:,end)));
DVBedroomNew = DVBedroomNew (:,2:end);

%based on removal of outlier for bathroom  dummyvar
DVBathroomNew = dummyvar(nominal(filenew.bathrooms(:,end)));
DVBathroomNew = DVBathroomNew (:,2:end);

%based on removal of outlier for Grade dummyvar
DVGradeNew = dummyvar(nominal(filenew.grade(:,end)));
DVGradeNew = DVGradeNew (:, 2:end);

%based on removal of outlier for Floors dummyvar
DVFloorsNew = dummyvar(nominal(filenew.floors(:,end)));
DVFloorsNew = DVFloorsNew (:, 2:end);

%based on removal of outlier for Condition dummyvar
DVConditionNew = dummyvar(nominal(filenew.condition(:,end)));
DVConditionNew = DVConditionNew (:, 2:end);

%based on removal of outlier for yearbuilt dummyvar
DVYrbuiltNew = dummyvar(nominal(filenew.yr_built(:,end)));
DVYrbuiltNew = DVYrbuiltNew (:, 2:end);

%based on removal of outlier for year renovated dummyvar
DVYrrenovatedNew = dummyvar(nominal(filenew.yr_renovated(:,end)));
DVYrrenovatedNew = DVYrrenovatedNew (:,2:end);

%based on removal of outlier for zipcode dummyvar
DVZipcodeNew = dummyvar(nominal(filenew.zipcode(:,end)));
DVZipcodeNew = DVZipcodeNew(:, 2:end);

%based on removal of outlier for view dummyvar
DVViewNew = dummyvar(nominal(filenew.view(:,end)));
DVViewNew = DVViewNew (:,2:end);

DVWaterfrontNew = dummyvar(nominal(filenew.waterfront(:,end)));
DVWaterfrontNew = DVWaterfrontNew (:,2:end);

%New x and y based on removing outliers
XcatNew = [DVBedroomNew, DVBathroomNew, DVFloorsNew, DVViewNew , DVConditionNew,DVGradeNew, DVYrbuiltNew, DVYrrenovatedNew, DVZipcodeNew, DVWaterfrontNew];
XvarNew = zscore([filenew.sqft_living filenew.sqft_basement filenew.lat filenew.long filenew.sqft_lot]);
XfinalNew = [XvarNew XcatNew];
Ynew = filenew.price;
md2 = fitlm (XfinalNew, Ynew)

opt = statset('UseParallel', true);

tic, [b fitinfo] = lasso(XfinalNew, Ynew, 'CV',10, 'Alpha', 1); toc
lassoPlot(b,fitinfo,'PlotType', 'Lambda', 'XScale', 'log')


%===============================================
%--TESTING OUT VARIOUS LAMBDAS FOR BEST MSE-----
%===============================================

B0 = b(:,82); %index of the b table 
nonzeros = sum(B0 ~= 0) % no zeros 
predictors = find(B0); % indices of nonzero predictors
md3 = fitlm(XfinalNew,Ynew,'PredictorVars',predictors)

B2 = b(:,70); %index of the b table 
nonzeros = sum(B2 ~= 0) % no zeros 
predictors2 = find(B2); % indices of nonzero predictors
md4 = fitlm(XfinalNew,Ynew,'PredictorVars',predictors2)

B3 = b(:,55); %index of the b table 
nonzeros = sum(B3 ~= 0) % no zeros 
predictors3 = find(B3); % indices of nonzero predictors
md5 = fitlm(XfinalNew,Ynew,'PredictorVars',predictors3)

B4 = b(:,40); %index of the b table 
nonzeros = sum(B4 ~= 0) % no zeros 
predictors4 = find(B4); % indices of nonzero predictors
md6 = fitlm(XfinalNew,Ynew,'PredictorVars',predictors4)

B5 = b(:,45); %index of the b table 
nonzeros = sum(B5 ~= 0) % no zeros 
predictors5 = find(B5); % indices of nonzero predictors
md7 = fitlm(XfinalNew,Ynew,'PredictorVars',predictors5)

B6 = b(:,60); %index of the b table 
nonzeros = sum(B6 ~= 0) % no zeros 
predictors6 = find(B6); % indices of nonzero predictors
md8 = fitlm(XfinalNew,Ynew,'PredictorVars',predictors6)

B7 = b(:,65); %index of the b table 
nonzeros = sum(B7 ~= 0) % no zeros 
predictors7 = find(B7); % indices of nonzero predictors
md9 = fitlm(XfinalNew,Ynew,'PredictorVars',predictors7)

B8 = b(:,50); %index of the b table 
nonzeros = sum(B8 ~= 0) % no zeros 
predictors8 = find(B8); % indices of nonzero predictors
md10 = fitlm(XfinalNew,Ynew,'PredictorVars',predictors8)

B9 = b(:,49); %index of the b table 
nonzeros = sum(B9 ~= 0) % no zeros 
predictors9 = find(B9); % indices of nonzero predictors
md11 = fitlm(XfinalNew,Ynew,'PredictorVars',predictors9)

B10 = b(:,56); %index of the b table 
nonzeros = sum(B10 ~= 0) % no zeros 
predictors10 = find(B10); % indices of nonzero predictors
md12 = fitlm(XfinalNew,Ynew,'PredictorVars',predictors10)

B11 = b(:,54); %index of the b table 
nonzeros = sum(B11 ~= 0) % no zeros 
predictors11 = find(B11); % indices of nonzero predictors
md13 = fitlm(XfinalNew,Ynew,'PredictorVars',predictors11)

B12 = b(:,30); %index of the b table 
nonzeros = sum(B12 ~= 0) % no zeros 
predictors12 = find(B12); % indices of nonzero predictors
md14 = fitlm(XfinalNew,Ynew,'PredictorVars',predictors12)

B13 = b(:,lambdaindex); %index of the b table 
nonzeros = sum(B13 ~= 0) % no zeros 
predictors13 = find(B13); % indices of nonzero predictors
md15 = fitlm(XfinalNew,Ynew,'PredictorVars',predictors13)


%============================
%--SVM MODEL SECTION---------
%============================


% Create price 4 labels for price category to perform logistic regression using SVM
PriceRange =filenew.price;
PriceRange(PriceRange<321950) =0;
PriceRange(PriceRange>=321950 & PriceRange<450000)=1;
PriceRange(PriceRange>=450000 & PriceRange<645000)=2;
PriceRange(PriceRange>=645000)=3;

% XcatNew is a concatenated variable with all the dummy variables of categorical predictors
XcatNew = [DVBedroomNew, DVBathroomNew, DVFloorsNew, DVViewNew , DVConditionNew,DVGradeNew, DVYrbuiltNew, DVYrrenovatedNew, DVZipcodeNew, DVWaterfrontNew];

% XvarNew is the concatenated variable with all the continous variable zscored together
XvarNew = zscore([filenew.sqft_living filenew.sqft_basement filenew.lat filenew.long filenew.sqft_lot]);

%XfinalNew is the concatenated variable of XvarNew and XcatNew
XfinalNew = [XvarNew XcatNew];

%X is only all the records of only those predictors which came out to be important ones after doing the Lasso
X=XfinalNew(:,[1 2 3 5 7 9 10 18 20 21 23 24 25 26 27 28 30 32 33 34 35 36 37 38 39 41 42 43 44 46 47 48 49 54 56 58 59 60 62 65 70 71 72 75 77 82 85 86 87 88 89 90 98 104 115 127 128 139 147 154 155 162 163 169 175 176 177 181 184 185 186 187 188 189 190 191 192 194 195 199 200 201 202 203 204 205 206 207 208 209 211 212 213 214 215 216 217 218 220 221 222 224 226 227 228 229 230 231 232 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 258 260 261 262 263 264 265 266]);

% Y is the price labels that we have created above from the price predictor.
Y_SVM=PriceRange; 

%build the model without kernel function
t = templateSVM('SaveSupportVectors',true);
SVM_MDL = fitcecoc(X,Y_SVM,'Coding','onevsall','Learners',t,'CrossVal','on');
loss=kfoldLoss(SVM_MDL,'Mode','individual');
predict_svm=predict(SVM_MDL.Trained{2,1},X);

%build the model with kernel rbf 
t = templateSVM('SaveSupportVectors',true,'KernelFunction','rbf');
SVM_MDL_RBF = fitcecoc(X,Y_SVM,'Coding','onevsall','Learners',t,'CrossVal','on');
cverror_individual = kfoldLoss(SVM_MDL_RBF,'Mode','individual');
predict_rbf=predict(SVM_MDL_RBF.Trained{5,1},X);

%build the model with kernel rbf and kernelscale auto
t = templateSVM('SaveSupportVectors',true,'KernelFunction','rbf','KernelScale','auto');
SVM_MDL_RBF_SCALE = fitcecoc(X,Y_SVM,'Coding','onevsall','Learners',t,'CrossVal','on');
cverror_individual = kfoldLoss(SVM_MDL_RBF_SCALE,'Mode','individual');
predict_rbf=predict(SVM_MDL_RBF_SCALE.Trained{8,1},X);

%build the model with kernel rbf box constraint 5
t = templateSVM('SaveSupportVectors',true, 'KernelFunction','rbf','BoxConstraint',5,'KernelScale','auto');
SVM_MDL_RBF_BOX = fitcecoc(X,Y_SVM,'Coding','onevsall','Learners',t,'CrossVal','on');
cverror_individual = kfoldLoss(SVM_MDL_RBF_BOX,'Mode','individual');
predict_rbf_box=predict(SVM_MDL_RBF_BOX.Trained{8,1},X);

%box constraint and kernel scale with c=0.000001
t = templateSVM('SaveSupportVectors',true, 'KernelFunction','gaussian','BoxConstraint',0.000001,'KernelScale','auto');
SVM_MDL_vLow = fitcecoc(X,Y_SVM,'Coding','onevsall','Learners',t,'CrossVal','on');
cverror_individual = kfoldLoss(SVM_MDL_vLow,'Mode','individual');
predict_vlow=predict(SVM_MDL_vLow.Trained{2,1},X);

%box constraint and kernel scale with c=1
t = templateSVM('SaveSupportVectors',true, 'KernelFunction','gaussian','BoxConstraint',1,'KernelScale','auto');
SVM_MDL1 = fitcecoc(X,Y_SVM,'Coding','onevsall','Learners',t,'CrossVal','on');
cverror_individual = kfoldLoss(SVM_MDL1,'Mode','individual');
predict_1=predict(SVM_MDL1.Trained{3,1},X);

%box constraint and kernel scale with c=2.5
t = templateSVM('SaveSupportVectors',true, 'KernelFunction','gaussian','BoxConstraint',2.5,'KernelScale','auto');
SVM_MDL2pt5 = fitcecoc(X,Y_SVM,'Coding','onevsall','Learners',t,'CrossVal','on');
cverror_individual = kfoldLoss(SVM_MDL2pt5,'Mode','individual');
predict_2pt5=predict(SVM_MDL5.Trained{5,1},X);

%box constraint and kernel scale with C=3
t = templateSVM('SaveSupportVectors',true, 'KernelFunction','gaussian','BoxConstraint',3,'KernelScale','auto');
SVM_MDL3 = fitcecoc(X,Y_SVM,'Coding','onevsall','Learners',t,'CrossVal','on');
predict_3=predict(SVM_MDL3.Trained{8,1},X);

%box constraint and kernel scale with C=3.5
t = templateSVM('SaveSupportVectors',true, 'KernelFunction','gaussian','BoxConstraint',3.5,'KernelScale','auto');
SVM_MDL3pt5 = fitcecoc(X,Y_SVM,'Coding','onevsall','Learners',t,'CrossVal','on');
cverror_individual = kfoldLoss(SVM_MDL3pt5,'Mode','individual');
predict_3pt5=predict(SVM_MDL3pt5.Trained{4,1},X);

%box constraint and kernel scale with C=4
t = templateSVM('SaveSupportVectors',true, 'KernelFunction','gaussian','BoxConstraint',4,'KernelScale','auto');
SVM_MDL4 = fitcecoc(X,Y_SVM,'Coding','onevsall','Learners',t,'CrossVal','on');
cverror_individual = kfoldLoss(SVM_MDL4,'Mode','individual');
predict_4=predict(SVM_MDL4.Trained{2,1},X);

%box constraint and kernel scale with C=4.5
t = templateSVM('SaveSupportVectors',true, 'KernelFunction','rbf','BoxConstraint',4.5,'KernelScale','auto');
SVM_MDL4pt5 = fitcecoc(X,Y_SVM,'Coding','onevsall','Learners',t,'CrossVal','on');
cverror_individual = kfoldLoss(SVM_MDL4pt5,'Mode','individual');
predict_4pt5=predict(SVM_MDL4pt5.Trained{7,1},X);

%box constraint and kernel scale with C=5
t = templateSVM('SaveSupportVectors',true, 'KernelFunction','gaussian','BoxConstraint',5,'KernelScale','auto');
SVM_MDL5 = fitcecoc(X,Y_SVM,'Coding','onevsall','Learners',t,'CrossVal','on');
cverror_individual = kfoldLoss(SVM_MDL5,'Mode','individual');
predict_5=predict(SVM_MDL5.Trained{,1},X);

%box constraint and kernel scale with C=10
t = templateSVM('SaveSupportVectors',true, 'KernelFunction','gaussian','BoxConstraint',10,'KernelScale','auto');
SVM_MDL10 = fitcecoc(X,Y_SVM,'Coding','onevsall','Learners',t,'CrossVal','on');
cverror_individual = kfoldLoss(SVM_MDL10,'Mode','individual');
predict_10=predict(SVM_MDL10.Trained{3,1},X);

%box constraint and kernel scale with C=25
t = templateSVM('SaveSupportVectors',true, 'KernelFunction','gaussian','BoxConstraint',25,'KernelScale','auto');
SVM_MDL25 = fitcecoc(X,Y_SVM,'Coding','onevsall','Learners',t,'CrossVal','on');
cverror_individual = kfoldLoss(SVM_MDL25,'Mode','individual');
predict_25=predict(SVM_MDL25.Trained{7,1},X);

%build the model with no kernel and c=25
t = templateSVM('SaveSupportVectors',true, 'BoxConstraint',25);
SVM_MDL25_Nok = fitcecoc(X,Y_SVM,'Coding','onevsall','Learners',t,'CrossVal','on');
cverror_individual = kfoldLoss(SVM_MDL25_Nok,'Mode','individual');
predict_25Nok=predict(SVM_MDL25_Nok.Trained{2,1},X);

%plot CFM for models
isLabels = unique(Y_SVM);
nLabels = numel(isLabels)
[n,p] = size(X)
[~,grpOOF] = ismember(predict_25Nok,isLabels); 
oofLabelMat = zeros(nLabels,n); 
idxLinear = sub2ind([nLabels n],grpOOF,(1:n)'); 
oofLabelMat(idxLinear) = 1; % Flags the row corresponding to the class 
[~,grpY] = ismember(Y_SVM,isLabels); 
YMat = zeros(nLabels,n); 
idxLinearY = sub2ind([nLabels n],grpY,(1:n)'); 
YMat(idxLinearY) = 1; 

figure;
plotconfusion(YMat,oofLabelMat);
h = gca;
h.XTickLabel = [num2cell(isLabels); {''}];
h.YTickLabel = [num2cell(isLabels); {''}];



%create ROC for all the four price labels

[label,NegLoss,score] = predict(SVM_MDL25_Nok.Trained{2,1},X);
hold on
[xpos, ypos, T, AUC] = perfcurve(Y_SVM, score(:,1),0);
plot(xpos, ypos) 
xlabel('False positive rate')
ylabel('True positive rate')
title('ROC for label 0')

[xpos, ypos, T, AUC] = perfcurve(Y_SVM, score(:,2),1);
plot(xpos, ypos) 
xlabel('False positive rate')
ylabel('True positive rate')
title('ROC for label 1')

[xpos, ypos, T, AUC] = perfcurve(Y_SVM, score(:,3),2);
plot(xpos, ypos) 
xlabel('False positive rate')
ylabel('True positive rate')
title('ROC for label 2')

[xpos, ypos, T, AUC] = perfcurve(Y_SVM, score(:,4),3);
plot(xpos, ypos) 
xlabel('False positive rate')
ylabel('True positive rate')
title('ROC for label 3')

%===========================
%--NN PATTERNNET MODEL-----
%===========================

%--Create Price Range for NN with NEW data--
PriceRange = price;
PriceRange(PriceRange<321950) =1;
PriceRange(PriceRange>=321950 & PriceRange<450000)=2;
PriceRange(PriceRange>=450000 & PriceRange<645000)=3;
PriceRange(PriceRange>=645000)=4;

%testing price ranges - TRIED VARIOUS MODELS, THIS IS AN EXAMPLE
%NOT USED FOR FINAL MODEL, STUCK WITH 4 RANGES
PriceRange = price;
PriceRange(PriceRange<300000) =1;
PriceRange(PriceRange>=300000 & PriceRange<350000)=2;
PriceRange(PriceRange>=350000 & PriceRange<400000)=3;
PriceRange(PriceRange>=400000 & PriceRange<550000)=4;
PriceRange(PriceRange>=550000 & PriceRange<600000)=5;
PriceRange(PriceRange>=600000)=6;

D_PriceRange = dummyvar(nominal(PriceRange)); 

% Create X & Y Training & test data 
xPattern = XfinalNew;
xPattern = xPattern.';
yPattern = D_PriceRange;
yPattern = yPattern.';

%==================SETTING UP NEURAL NETWORKS & PARAMETERS ==========%
ffn1 = patternnet([5 3]); %Neural Network with 1 neuron
ffn1.trainParam.goal = 1e-20;
ffn1.trainParam.min_grad = 1e-010;
ffn1.trainParam.epochs=2000;            %Epochs changed to 2000
ffn1.trainFcn = 'trainscg';             %Change to Scaled Conjugate Gradient
ffn1.layers{1}.transferFcn='poslin';    %change to ReLU transfer function

%Train NN
ffn1 = train(ffn1, xPattern, yPattern); 
yhat = sim(ffn1, xPattern);
[xpos, ypos, T, AUC] = perfcurve(yACTUAL,yhat10, '1');
[xpos, ypos, T, AUC] = perfcurve(y_actual,, '1');
figure, plot(xpos, ypos)
xlim([-0.01 1.01]), ylim([-0.01 1.01])
xlabel('\bf False Positive Rate'), ylabel('\bf True Positive Rate')
title('\bf ROC for Interesting (1) Classification with 1 Neuron')
legend('Interesting (1)')

% --- Feedforward NN ----------------------------------------------
%--Tested out but the running time was very slow, so we did not use
%------------------------------------------------------------------
% Create X & Y Training & test data (19211 Training, 10% of new dataset is Test 2135)
xTest = XfinalNew(19212: end, :);
xTest = xTest.';
xTrain =XfinalNew(1:19211, :);
xTrain = xTrain.';
yTest = price_FinalD2(19212: end, :);
yTest = yTest.';
yTrain = price_FinalD2(1:19211, :);
yTrain = yTrain.';

net = feedforwardnet(5);
net.trainParam.goal = 1e-20;
net.trainParam.min_grad = 1e-010;
net.trainParam.epochs=2000;            %Epochs changed to 2000
net.trainFcn = 'trainscg';             %Change to Scaled Conjugate Gradient
%ffn1.layers{1}.transferFcn='poslin';  

net = train(net,xTrain, yTrain);
yHat = sim(net, xTest);
plotroc(yHat, yTest);


%============================
%--NN AUTO ENCODER MODEL-----
%============================

%--Create Price Range for NN with NEW data--
PriceRange = price;
PriceRange(PriceRange<321950) =1;
PriceRange(PriceRange>=321950 & PriceRange<450000)=2;
PriceRange(PriceRange>=450000 & PriceRange<645000)=3;
PriceRange(PriceRange>=645000)=4;

D_PriceRange = dummyvar(nominal(PriceRange)); 


% --- Auto Encoder Neuro Network ----
%------------------------------------%
% Create X & Y Training & test data (19211 Training, 10% of new dataset is Test 2135)
xTrain = XfinalNew(1:19211, :);
xTrain = xTrain.';
xTest = XfinalNew(19212:end, :);
xTest = xTest.';
yTrain = D_PriceRange(1:19211, :);
yTrain = yTrain.';
yTest =  D_PriceRange(19212:end, :);
yTest = yTest.';

autoenc1 = trainAutoencoder(xTrain,100 , 'MaxEpochs', 2000, 'SparsityProportion',0.1, 'ScaleData', false);
feat1 = encode(autoenc1, xTrain);   %stoped at 1000 epochs

autoenc2 = trainAutoencoder(feat1, 50, 'MaxEpochs', 1000, 'SparsityProportion',0.1, 'ScaleData', false);
feat2 = encode(autoenc2, feat1);

autoenc3 = trainAutoencoder(feat2, 25, 'MaxEpochs', 700, 'SparsityProportion',0.1, 'ScaleData', false);
feat3 = encode(autoenc3, feat2);

%USED THROUGHOUT TRIALS, NOT IN FINAL  MODEL
autoenc4 = trainAutoencoder(feat3, 10, 'MaxEpochs', 500, 'SparsityProportion',0.1, 'ScaleData', false);
feat4 = encode(autoenc4, feat3);

%USED THROUGHOUT TRIALS, NOT IN FINAL MODEL
autoenc5 = trainAutoencoder(feat4, 50, 'MaxEpochs', 300, 'SparsityProportion',0.1, 'ScaleData', false);
feat5 = encode(autoenc5, feat4);

%USED THROUGHOUT TRIALS, NOT IN FINAL MODEL
autoenc6 = trainAutoencoder(feat5, 10, 'MaxEpochs', 300, 'SparsityProportion',0.1, 'ScaleData', false);
feat6 = encode(autoenc6, feat5);

%USED THROUGHOUT TRIALS, NOT IN FINAL MODEL
autoenc7 = trainAutoencoder(feat6, 10, 'MaxEpochs', 300, 'SparsityProportion',0.1, 'ScaleData', false);
feat7 = encode(autoenc7, feat6);

softnet = trainSoftmaxLayer(feat3, yTrain, 'MaxEpochs', 700);

deepnet = stack(autoenc1, autoenc2, autoenc3, softnet);
y_NN2 = deepnet(xTest);
plotconfusion(yTest, y_NN2);

%Fine Tuning
deepnet = train(deepnet, xTrain, yTrain);
y_NN2 = deepnet(xTest);
plotconfusion(yTest, y_NN2);
plotroc(yTest, y_NN2);
