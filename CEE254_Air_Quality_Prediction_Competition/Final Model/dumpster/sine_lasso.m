%% 5. sine_lasso
function [pred_pm2d5,rmse] = sine_lasso(train_data_sampled,test_CV,test_data,t0,problem_type)

% Periodic Fit Time + other fits for other vars
% shrinkage method = ridge

t=minutes(test_data.time-t0);
if problem_type == 1
    % expand to all 180 mins within the 3-hr prediction duration
    for i=1:length(t)
        t_expand((60*i-59):60*i,1) =(t(i)-29):1:(t(i)+30);
        lat_expand((60*i-59):60*i,1) = test_data.lat(i);
        lon_expand((60*i-59):60*i,1) = test_data.lon(i);
    end
elseif problem_type == 2
    % expand to all 24*60 mins within the 24-hr prediction duration
    for i=1:length(t)
        t_expand((60*i-59):60*i,1) =(t(i)-29):1:(t(i)+30);
        lat_expand((60*i-59):60*i,1) = test_data.lat(i);
        lon_expand((60*i-59):60*i,1) = test_data.lon(i);
    end
elseif problem_type == 3
    % expand to all 60 mins within the 1-hr prediction duration
    for i=1:length(t)
        t_expand((5*i-4):5*i,1) =(t(i)-2):1:(t(i)+2);
        lat_expand((60*i-59):60*i,1) = test_data.lat(i);
        lon_expand((60*i-59):60*i,1) = test_data.lon(i);
    end
else
    fprintf("Please enter the correct problem type! ")
end
% depending on the probelm type, the tuned lambda might be different
lam = [10,10,10];
ome = [1/525600,1/131400,1/10080,1/360,1/60,1/20,1/10;...
    1/525600,1/131400,1/1440,1/720,1/360,1/60,1/20;...
    1/525600,1/131400,1/10080,1/1440,1/720,1/360,1/60]; 
% Frequency content Yearly, 3 Month, 1 week, 1 day, 12 hrs, 6 hrs, 1hr   1/525600,1/131400,1/10080,1/1440,1/720,1/360,1/60]; % Frequency content Yearly, 3 Month, 1 week, 1 day, 12 hrs, 6 hrs, 1hr
% lambda is tuned such that it results in smallest average cost of 100 samples
lambda = lam(problem_type);
omega = ome(problem_type,:); % Frequency content Yearly, 3 Month, 1 week, 1 day, 12 hrs, 6 hrs, 1hr
X = [ones(length(train_data_sampled(:,1)),1),sin(train_data_sampled(:,1)*omega),cos(train_data_sampled(:,1)*omega),...
    train_data_sampled(:,6),...
    train_data_sampled(:,6).^2, train_data_sampled(:,6).^3, ...
    train_data_sampled(:,7), train_data_sampled(:,7).^2, ...
    train_data_sampled(:,7).^3];
dim = size(X,2);
[beta, FitInfo] = lasso(X(:,2:dim), train_data_sampled(:,5), 'Lambda',lambda);
intercept = FitInfo.Intercept;
pred_pm2d5_full = [ones(length(t_expand),1),sin(t_expand*omega),cos(t_expand*omega),...
    lat_expand,lat_expand.^2, lat_expand.^3, ...
    lon_expand, lon_expand.^2, ...
    lon_expand.^3]*[intercept;beta];
pm2d5_CV = [ones(length(test_CV),1),sin(test_CV(:,1)*omega),cos(test_CV(:,1)*omega),...
    test_CV(:,6),test_CV(:,6).^2, test_CV(:,6).^3, ...
    test_CV(:,7), test_CV(:,7).^2, test_CV(:,7).^3]*[intercept;beta];
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