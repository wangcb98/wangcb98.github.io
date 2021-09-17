function [pm2d5_pred,rmse] = crossVal(fun,train_data_sampled,test_data,t0,problem_type)
% Perform 5-fold cross validation. 
% Return best model and associated error.
indx = rand(length(train_data_sampled),1);
pm2d5_pred = zeros(height(test_data),1);
rmse = inf;
for i = 1:5
    ind = and(indx(:,1)>=(i-1)/5,indx(:,1)<i/5);
    train = train_data_sampled(~ind,:);
    test_CV = train_data_sampled(ind,:);
    [pm2d5,err] = fun(train,test_CV,test_data,t0,problem_type);
    if err<rmse
        pm2d5_pred = pm2d5;
        rmse = err;
    end
end
end