
function [pm2d5_pred,rmse] = futureVal(fun,train_data_sampled,test_data,t0,problem_type)
% Perform future validation to estimate  model's capacity to forcast PM2.5
% measurements. Function uses the initial 80% of the time data to build a
% model and the remaining 20% to test the extrapolation error. Returns
% pm2d5 prediction for a model built on the entire dataset (error IS NOT
% associated directly with the provided model but relative proportions can
% guide model weights.)
t_lim = 0.8*(max(train_data_sampled(:,1))-min(train_data_sampled(:,1)));
train = train_data_sampled(train_data_sampled(:,1)<t_lim,:);
test_FV = train_data_sampled(train_data_sampled(:,1)>=t_lim,:);
[~,rmse] = fun(train,test_FV,test_data,t0,problem_type);
[pm2d5_pred,~] = fun(train_data_sampled,test_FV,test_data,t0,problem_type);
end
