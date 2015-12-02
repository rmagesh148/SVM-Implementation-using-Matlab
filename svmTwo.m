%# Initializing labels 
Labels=ones(10000,1); 
Labels([1:500,1001:1500,2001:2500,3001:3500,4001:4500,5001:5500,6001:6500,7001:7500 ,8001:8500,9001:9500])=2; 
%# Reading Input 
dataInput=load('data.txt'); 
%# Kfold validation 
countForKFold=10; 
crossValidationValue = crossvalind('Kfold',Labels, k); 
classValue = classperf(Labels); 
%# SVM Training with Polynomial Kernel Function of degree 3 
for i = 1:
	countForKFold testIndiciesValue = (crossValidationValue == i); 
	trainIndiciesValue = ~testIndiciesValue; 
	svmModel = fitcsvm( dataInput(trainIndiciesValue,:), Labels(trainIndiciesValue),'BoxConstraint',2e-1, 'KernelFunction','polynomial', 'PolynomialOrder',3); 
	pred = predict(svmModel, dataInput(testIndiciesValue,:)); 
	classValue = classperf(classValue, pred, testIndiciesValue); 
end 
%# accuracy 
classValue.CorrectRate 
%# confusion matrix 
classValue.CountingMatrix
