function [pm2d5_pred,rmse] = crossVal(fun,train_data_sampled,test_data,t0,problem_type)
    indx = rand(length(train_data_sampled),5)<0.8;
    pm2d5_pred = zeros(height(test_data),1);
    rmse = inf;
    for i = 1:5
        train = train_data_sampled(indx(:,i),:);
        test_CV = train_data_sampled(~indx(:,i),:);
        [pm2d5,err] = fun(train,test_CV,test_data,t0,problem_type);
        if err<rmse
            pm2d5_pred = pm2d5;
            rmse = err;
        end
    end

end
