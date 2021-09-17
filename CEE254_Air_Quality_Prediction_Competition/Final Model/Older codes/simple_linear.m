function [pm2d5_pred,rmse] = simple_linear(train_data_sampled,test_CV,test_data,t0,problem_type)
%performs multivaraite linear regression

beta = mvregress(train_data_sampled(:,[1,2,3,4,6,7]),train_data_sampled(:,5));
test_data.minute=minutes(test_data.time-t0);
pm2d5_pred = table2array(test_data(:,[7,2,3,4,5,6]))*beta;
pm2d5_CV = test_CV(:,[1,2,3,4,6,7])*beta;
Residual = (test_CV(:,5)-pm2d5_CV)';
rmse = sqrt(mean(Residual.*Residual));

end