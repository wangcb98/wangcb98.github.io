function [pred_pm2d5,rmse] = sine_lasso(train_data_sampled,test_CV,test_data,t0,problem_type)

% Periodic Fit Time + other fits for other vars
% shrinkage method = ridge

t=minutes(test_data.time-t0);
if problem_type == 1
    % expand to all 180 mins within the 3-hr prediction duration
    for i=1:length(t)
        t_expand((60*i-59):60*i,1) =(t(i)-29):1:(t(i)+30);
    end
elseif problem_type == 2
    % expand to all 24*60 mins within the 24-hr prediction duration
    for i=1:length(t)
        t_expand((60*i-59):60*i,1) =(t(i)-29):1:(t(i)+30);
    end
elseif problem_type == 3
    % expand to all 60 mins within the 1-hr prediction duration
    for i=1:length(t)
        t_expand((5*i-4):5*i,1) =(t(i)-2):1:(t(i)+2);
    end  
else
    fprintf("Please enter the correct problem type! ")
end
% depending on the probelm type, the tuned lambda might be different
lam = [100,100,100];
% lambda is tuned such that it results in smallest average cost of 100 samples
lambda = lam(problem_type);
omega = [1/525600,1/131400,1/10080,1/1440,1/720,1/360,1/60]; % Frequency content Yearly, 3 Month, 1 week, 1 day, 12 hrs, 6 hrs, 1hr
X = [ones(length(train_data_sampled(:,1)),1),sin(train_data_sampled(:,1)*omega),cos(train_data_sampled(:,1)*omega)];
dim = size(X,2);
[beta, FitInfo] = lasso(X(:,2:dim), train_data_sampled(:,5), 'Lambda',lambda);
intercept = FitInfo.Intercept;
pred_pm2d5_full = [ones(length(t_expand),1),sin(t_expand*omega),cos(t_expand*omega)]*[intercept;beta];
pm2d5_CV = [ones(length(test_CV),1),sin(test_CV(:,1)*omega),cos(test_CV(:,1)*omega)]*[intercept;beta];
% compute the hourly mean
n = length(t_expand)/length(t); % Number of elements to compute the mean over
s1 = size(pred_pm2d5_full, 1);  % Find the next smaller multiple of n
m  = s1 - mod(s1, n);
pred_pm2d5_reshape  = reshape(pred_pm2d5_full(1:m), n, []);     % Reshape x to a [n, m/n] matrix
pred_pm2d5 = transpose(sum(pred_pm2d5_reshape, 1) / n);
Residual = pm2d5_CV-test_CV(:,5);
residual_square = Residual.*Residual;
aux_beta_abs = abs(beta);
cost = sum(residual_square(:)) + lambda*sum(aux_beta_abs(:));
rmse = sqrt(mean(residual_square));
RMSE = [rmse,cost];%return both RMSE and cost
end