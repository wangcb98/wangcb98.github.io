%% 2. simple_linear_lasso
function [pm2d5_pred,rmse] = simple_linear_lasso(train_data_sampled,test_CV,test_data,t0,problem_type)

% Use lasso to select features (time, hmd, spd, tmp, lat, lon)

% depending on the probelm type, the tuned lambda is different
lam = [10,50,50];
X_train = train_data_sampled(:,[1,2,3,4,6,7]);
Y_train = train_data_sampled(:,5);
test_data.minute=minutes(test_data.time-t0);
test_data_mat = table2array(test_data(:,[7,2,3,4,5,6]));
X_test = [ones(length(test_data_mat(:,1)),1),test_data_mat];
dim = size(X_train,2);
lambda = lam(problem_type); % chosen by cross validation
[beta, FitInfo] = lasso(X_train(:,2:dim), Y_train, 'Lambda',lambda);
intercept = FitInfo.Intercept;
pm2d5_pred = X_test(:,2:dim)*beta + intercept*ones(length(X_test(:,1)),1);
X_test_CV = [ones(length(test_CV(:,1)),1),test_CV(:,[1,2,3,4,6,7])];
pm2d5_CV = X_test_CV(:,2:dim)*beta + intercept*ones(length(X_test_CV(:,1)),1);
residual_square = (pm2d5_CV-test_CV(:,5)).*(pm2d5_CV-test_CV(:,5));
aux_beta_abs = abs(beta);
cost = sum(residual_square(:)) + lambda*sum(aux_beta_abs(:));
rmse = sqrt(mean(residual_square));
RMSE = [rmse,cost];

end
