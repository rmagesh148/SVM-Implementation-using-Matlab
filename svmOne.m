%# Initializing labels 
Labels(1:5000)=0; 
Labels(5001:10000)=1; 

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