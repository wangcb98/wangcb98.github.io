%% Main body
function [pred_pm2d5] = pm2d5_pred_model(train_data, test_data, problem_type)
% pm2d5_pred_model uses provided training data to estimate or forecast
% PM2.5 air pollution values at a desired time and location. The input and
% output data shall be of the form specified below:
% train_data = Px7 matrix of training data with time, location, other data.
% test_data = Mx6 matrix, where M is size of output
% predictionType = desired prediction mode, options include:
% (1) : short term, (2) : long term, (3) : interpolation 
% pred_pm2d5 = Mx1 output stored as a vector with pm2d5 predictions associated
% with desired location and time. 

% (1) GPR, SineLasso, and SimpleLinear performs relatively better than
% other methods, with RMSE at around 15.
% (2) SimpleLinearLasso and PolyLasso perfroms well (RMSE = 6~7), GPR
% performs okay (RMSE around 15). Others performs poorly.
% (3) GPR and SineRidge both performs well (RMSE<10).

%% The real "test_data" will be a M-by-6 matrix, be sure to change the code before submitting
%% testMSE can not be computed in "real-world" application
%%
% 
% CEE 254: Data Analytics 
% Chenbo Wang
% Ben Flanagan
% Fall 2020
%
%% Main Function 
tic
[train_data_sampled] = movAvg(train_data); % Time is in minutes
% fill in missing values using gpr model
[gprMdl,train_data_sampled,rmse_gpr] = gpr_fill_missing_vals(train_data_sampled);
t0 = train_data.time(1);

switch problem_type
    case 1 % Short Term
        % The RMSE = for 100 random samples using logic-tree-based model
        % The logic tree model comprises of:
        % SimpleLinear(30%), SineLasso(30%), and GPR(40%)
        % takes about 1~2 mins
        
        % Start implementing short-term prediction
        % Prediction methods include "SimpleLinear", "SimpleLinearLasso", "PolyLasso","SineRidge","SineLasso", and "GPR".
        %% 'SimpleLinear'
        % mean of RMSE = 15.0126 for 100 samplings. (std = 7.5479)
        % Use all six features (time, hmd, spd, tmp, lat, lon) as predictors
        [pred_pm2d5_simplelinear,rmse_simplelinear] = crossVal(@simple_linear,train_data_sampled,test_data,t0,1);
        
        %% 'SimpleLinearLasso'
        % mean of RMSE = 23.6520 for 100 samplings. (std = 6.1401)
        % Use lasso to select features (time, hmd, spd, tmp, lat, lon)
        [pred_pm2d5_simplelinearlasso,rmse_simplelinearlasso] = crossVal(@simple_linear_lasso,train_data_sampled,test_data,t0,1);
        
        %% 'PolyLasso'
        % mean of RMSE = 25.8857 for 100 samplings. (std = 6.0589)
        % degree = 3
        [pred_pm2d5_polylasso,rmse_polylasso] = crossVal(@poly_lasso,train_data_sampled,test_data,t0,1);
        
        %% 'SineRidge'
        % mean of RMSE = 25.8508 for 100 samplings. (std = 11.2325)
        % Periodic Fit Time + other fits for other vars
        [pred_pm2d5_sineridge,rmse_sineridge] = crossVal(@sine_ridge,train_data_sampled,test_data,t0,1);
        
        %% 'SineLasso'
        % mean of RMSE = 15.4855 for 100 samplings. (std = 10.9922)
        %% Periodic Fit Time + other fits for other vars
        [pred_pm2d5_sinelasso,rmse_sinelasso] = crossVal(@sine_lasso,train_data_sampled,test_data,t0,1);
        
        %% 'GPR'
        % mean of RMSE = 13.9624 for 100 samplings. (std = 8.8551)
        % Gaussian Process Regression
        % GPR model is previously built when interpolating missing values
        % We can conveniently take advantage of it
        t=minutes(test_data.time-t0);
        test_data.minute = t;
        [pred_pm2d5_gpr] = predict(gprMdl,table2array(test_data(:,[7,2,3,4,5,6])));
        % Residual = pred_pm2d5_gpr-test_data.pm2d5;
        % RMSE = sqrt(mean(Residual.*Residual));
        
        %% Decision tree       
        short_term_pred = [pred_pm2d5_simplelinear,pred_pm2d5_simplelinearlasso,...
            pred_pm2d5_polylasso,pred_pm2d5_sineridge,pred_pm2d5_sinelasso,...
            pred_pm2d5_gpr]
        weight = 1./([rmse_simplelinear;rmse_simplelinearlasso;...
            rmse_polylasso;rmse_sineridge;rmse_sinelasso;...
            rmse_gpr].^2);%[0.3;0;0;0;0.3;0.4];
        
        TF_preds = ismissing(short_term_pred);
        err_preds = or(short_term_pred<=0,short_term_pred>400);
        % return the row and col index of the missing values
        [~,col] = find(TF_preds);
        weight(col) = 0;
        [~,col] = find(err_preds);
        weight(col) = 0;
%         if sum(weight(:)) < 0.99
%             fprintf("NAN occurs in predictions! Please check!")
%             % reweight according to available branches
%             
%         end
        weight = weight ./sum(weight(:))
        short_term_pred_clean = short_term_pred;
        short_term_pred_clean(:,col) = 0;
        pred_pm2d5 = short_term_pred_clean*weight;
                 
    case 2 % Long Term
        % The RMSE = for 100 random samples using logic-tree-based model
        % The logic tree model comprises of：
        % SimpleLinearLasso（40%）, PolyLasso(40%), and GPR(20%)
        % takes about 10~15 mins
        
        % Start implementing long-term prediction
        % Prediction methods include "SimpleLinear", "SimpleLinearLasso", "PolyLasso","SineRidge","SineLasso", and "GPR".
        
        %% 'SimpleLinear'
        % mean of RMSE = 18.2755 for 100 samplings. (std = 7.7438)
        % Use all six features (time, hmd, spd, tmp, lat, lon) as predictors
        [pred_pm2d5_simplelinear,rmse_simplelinear] = crossVal(@simple_linear,train_data_sampled,test_data,t0,2);
        
        %% 'SimpleLinearLasso'
        % mean of RMSE = 6.0117 for 100 samplings. (std = 3.7183)
        % also performs quite well
        % Use LASSO to select features (time, hmd, spd, tmp, lat, lon)
        [pred_pm2d5_simplelinearlasso,rmse_simplelinearlasso] = crossVal(@simple_linear_lasso,train_data_sampled,test_data,t0,2);
        
        %% "PolyLasso"
        % mean of RMSE = 6.7530 for 100 samplings. (std = 3.9355)
        % use degree = 3
        [pred_pm2d5_polylasso,rmse_polylasso] = crossVal(@poly_lasso,train_data_sampled,test_data,t0,2);

        %% "SineRidge"
        % mean of RMSE = 19.1008 for 100 samplings. (std = 8.2232)
        % Periodic Fit Time + other fits for other vars
        [pred_pm2d5_sineridge,rmse_sineridge] = crossVal(@sine_ridge,train_data_sampled,test_data,t0,2);
               
        %% "SineLasso"
        % mean of RMSE = 21.0483 for 100 samplings. (std = 13.7163)
        % Periodic Fit Time + other fits for other vars
        [pred_pm2d5_sinelasso,rmse_sinelasso] = crossVal(@sine_lasso,train_data_sampled,test_data,t0,2);
                
        %% "GPR"
        % mean of RMSE = 15.2164 for 100 samplings. (std = 9.1975)
        % takes about 10 mins to run
        % Gaussian Process Regression
        
        % GPR model is previously built when interpolating missing values
        % We can conveniently take advantage of it
        t=minutes(test_data.time-t0);
        test_data.minute = t;
        [pred_pm2d5_gpr] = predict(gprMdl,table2array(test_data(:,[7,2,3,4,5,6])));
        % Residual = pred_pm2d5_gpr-test_data.pm2d5;
        % RMSE = sqrt(mean(Residual.*Residual));
        
        %% Decision tree
        long_term_pred = [pred_pm2d5_simplelinear,pred_pm2d5_simplelinearlasso,...
            pred_pm2d5_polylasso,pred_pm2d5_sineridge,pred_pm2d5_sinelasso,...
            pred_pm2d5_gpr]
        weight = 1./([rmse_simplelinear;rmse_simplelinearlasso;...
            rmse_polylasso;rmse_sineridge;rmse_sinelasso;...
            rmse_gpr]);%[0.3;0;0;0;0.3;0.4];
        
        TF_preds = ismissing(long_term_pred);
        err_preds = or(long_term_pred<=0,long_term_pred>400);
        % return the row and col index of the missing values
        [~,col] = find(TF_preds);
        weight(col) = 0;
        [~,col] = find(err_preds);
        weight(col) = 0;
%         if sum(weight(:)) < 0.99
%             fprintf("NAN occurs in predictions! Please check!")
%             % reweight according to available branches
%             
%         end
        weight = weight ./sum(weight(:))
        long_term_pred_clean = long_term_pred;
        long_term_pred_clean(:,col) = 0;
        pred_pm2d5 = long_term_pred_clean*weight;
        
        
        
    case 3 % Interpolation
        % The RMSE = for 100 random samples using logic-tree-based model
        % The logic tree model comprises of:
        % PolyLasso(0.35), SineRidge(0.3), and GPR(0.35)
        % takes about 3~4 mins
        
        %% "PolyLasso"
        % mean of RMSE = 9.4493 for 100 samplings. (std = 5.3550)
        [pred_pm2d5_polylasso,rmse_polylasso] = crossVal(@poly_lasso,train_data_sampled,test_data,t0,3);
        
        %% "SineRidge"
        % mean of RMSE = 9.9847 for 100 samplings. (std = 6.1161)
        [pred_pm2d5_sineridge,rmse_sineridge] = crossVal(@sine_ridge,train_data_sampled,test_data,t0,3); 

%         %% "SineLasso"
%         % mean of RMSE = 21.4065 for 100 samplings. (std = 14.5773)
%         % Too bad to be used
%         [pred_pm2d5_sinelasso] = sine_lasso(train_data_sampled,test_data,t0,3);
        
        %% "GPR"
        % mean of RMSE = 9.7534 for 100 samplings. (std = 7.4176)
        % Gaussian Process Regression
        
        % GPR model is previously built when interpolating missing values
        % We can conveniently take advantage of it
        t=minutes(test_data.time-t0);
        test_data.minute = t;
        [pred_pm2d5_gpr] = predict(gprMdl,table2array(test_data(:,[7,2,3,4,5,6])));
        % Residual = pred_pm2d5_gpr-test_data.pm2d5;
        % RMSE = sqrt(mean(Residual.*Residual));
        
        %% Decision tree
        interpo_pred = [pred_pm2d5_polylasso,pred_pm2d5_sineridge,pred_pm2d5_gpr]
        weight = 1./([rmse_polylasso;rmse_sineridge;rmse_gpr].^2);%[0.3;0;0;0;0.3;0.4];
        
        TF_preds = ismissing(interpo_pred);
        err_preds = or(interpo_pred<=0,interpo_pred>400);
        % return the row and col index of the missing values
        [~,col] = find(TF_preds);
        weight(col) = 0;
        [~,col] = find(err_preds);
        weight(col) = 0;
%         if sum(weight(:)) < 0.99
%             fprintf("NAN occurs in predictions! Please check!")
%             % reweight according to available branches
%             
%         end
        weight = weight ./sum(weight(:))
        interpo_pred_clean = interpo_pred;
        interpo_pred_clean(:,col) = 0;
        pred_pm2d5 = interpo_pred_clean*weight;
        
end

toc
end


%% Other functions called by the main function

%% movAvg
function [train_data_sampled] = movAvg(train_data)
% Function converts the input training data from 3 second samples to one
% minute average samples. Function divides input spatial coordinates into
% defined grid and assigns each point to one of the grid squares. Time
% averages are computed at each spatial square where data exists.
gridType = 'Cartesian'; % Options: 'Cartesian', 'Polar'
gridElements = 10; % nElements along one side of cartisian grid or rings of polar grid.
%timeStep = 1; % Time in minutes to group data.
switch gridType
    case 'Cartesian'
        train_data_sampled = [];
        lat_length = max(train_data.lat)-min(train_data.lat);
        lat_grid = ceil(((train_data.lat-min(train_data.lat))/lat_length)*(gridElements));
        lat_centroids = (min(train_data.lat)+lat_length/2/gridElements):(lat_length/gridElements):max(train_data.lat);
        lat_grid(lat_grid == 0) = 1;
        sampled_lat = lat_centroids(lat_grid);
        lon_length = max(train_data.lon)-min(train_data.lon);
        lon_grid = ceil(((train_data.lon-min(train_data.lon))/lon_length)*(gridElements));
        lon_centroids = (min(train_data.lon)+lon_length/2/gridElements):(lon_length/gridElements):max(train_data.lon);
        lon_grid(lon_grid == 0) = 1;
        sampled_lon = lon_centroids(lon_grid);
        train_time = round(minutes(train_data.time-min(train_data.time)),-1);
        M = [train_time,table2array(train_data(:,2:5)),sampled_lat',sampled_lon'];
        M = sortrows(M,[6,7,1]);
        m_indx = (2:length(M))';
        m_indx = m_indx(M(1:end-1,7) ~= M(2:end,7));
        c_indx = 1;
        for i = 1:length(m_indx)
            r = M(c_indx:m_indx(i)-1,:);
            [C,~,idx] = unique(r(:,1),'stable');
            val2 = accumarray(idx,r(:,2),[],@mean); 
            val3 = accumarray(idx,r(:,3),[],@mean);
            val4 = accumarray(idx,r(:,4),[],@mean);
            val5 = accumarray(idx,r(:,5),[],@mean);
            train_data_sampled = [train_data_sampled;C,val2,val3,val4,val5,repmat(M(c_indx,6:7),length(val2),1)];
            c_indx = m_indx(i);
        end
    case 'Polar'
end
end


%% gpr_fill_missing_vals
function [gprMdl,train_data_sampled,rmse] = gpr_fill_missing_vals(train_data_sampled)

%   check for missing value
TF = ismissing(train_data_sampled);
%   return the row and col index of the missing values
[row,col] = find(TF);
train_data_sample_clean = train_data_sampled(all(~isnan(train_data_sampled),2),:);
%   interpolate at missing values (if any)

% the kernel type is selected through multiple random sampling
train_interpo_x = train_data_sample_clean(:,[1,2,3,4,6,7]);
train_interpo_y = train_data_sample_clean(:,5);
gprMdl = fitrgp(train_interpo_x,train_interpo_y,'KernelFunction','matern32','CrossVal','on','KFold',5);
err = sqrt(kfoldLoss(gprMdl,'LossFun','mse','Mode','individual'));
bestMdl = 1:5;
bestMdl = bestMdl(err == min(err));
for i=1:length(row)
    test_interpo_sampled = train_data_sampled(row(i),:);
    test_interpo_x = test_interpo_sampled(:,[1,2,3,4,6,7]);
    % NEED TO FIX: the first column test_interpo_sampled is not in
    % (continue) datetime and the column name is missing
    train_data_sampled(row(i),col(i))=predict(gprMdl.Trained{bestMdl},test_interpo_x);
end
gprMdl = gprMdl.Trained{bestMdl};
TF_new = ismissing(train_data_sampled);
if sum(TF_new(:)) ~= 0
    fprintf("Something went wrong! Missing values still persist.")
    % if this happens, there probably are missing weather/time data
end
rmse = min(err);
end


%% simple_linear
function [pm2d5_pred,rmse] = simple_linear(train_data_sampled,test_CV,test_data,t0,problem_type)
%performs multivaraite linear regression

beta = mvregress(train_data_sampled(:,[1,2,3,4,6,7]),train_data_sampled(:,5));
test_data.minute=minutes(test_data.time-t0);
pm2d5_pred = table2array(test_data(:,[7,2,3,4,5,6]))*beta;
pm2d5_CV = test_CV(:,[1,2,3,4,6,7])*beta;
Residual = (test_CV(:,5)-pm2d5_CV)';
rmse = sqrt(mean(Residual.*Residual));

end


%% simple_linear_lasso
function [pm2d5_pred,rmse] = simple_linear_lasso(train_data_sampled,test_CV,test_data,t0,problem_type)

% Use lasso to select features (time, hmd, spd, tmp, lat, lon)

% depending on the probelm type, the tuned lambda is different
lam = [10,50];
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


%% sine_ridge
function [pred_pm2d5,rmse] = sine_ridge(train_data_sampled,test_CV,test_data,t0,problem_type)

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
lam = [0.1,0.1,0.01];
% lambda is tuned such that it results in smallest average cost of 100 samples
lambda = lam(problem_type);
% omega: frequency content Yearly, 3 Month, 1 week, 1 day, 12 hrs, 6 hrs, 1hr
omega = [1/525600,1/131400,1/10080,1/1440,1/720,1/360,1/60]; 
X = [ones(length(train_data_sampled(:,1)),1),sin(train_data_sampled(:,1)*omega),cos(train_data_sampled(:,1)*omega)];
beta = (X'*X+lambda*eye(size(X,2)))\(X'*train_data_sampled(:,5));
pred_pm2d5_full = [ones(length(t_expand),1),sin(t_expand*omega),cos(t_expand*omega)]*beta;
pm2d5_CV = [ones(length(test_CV),1),sin(test_CV(:,1)*omega),cos(test_CV(:,1)*omega)]*beta;
% compute the hourly mean
n = length(t_expand)/length(t); % Number of elements to compute the mean over
s1 = size(pred_pm2d5_full, 1);  % Find the next smaller multiple of n
m  = s1 - mod(s1, n);
pred_pm2d5_reshape  = reshape(pred_pm2d5_full(1:m), n, []);     % Reshape x to a [n, m/n] matrix
pred_pm2d5 = transpose(sum(pred_pm2d5_reshape, 1) / n);
Residual = pm2d5_CV-test_CV(:,5);
residual_square = Residual.*Residual;
aux_beta_square = beta.*beta;
cost = sum(residual_square(:)) + lambda*sum(aux_beta_square(2:size(X,2)));
rmse = sqrt(mean(residual_square));
RMSE = [rmse,cost];%return both RMSE and cost
end


%% sine_lasso
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


%% poly_lasso
function [pm2d5_pred,rmse] = poly_lasso(train_data_sampled,test_CV,test_data,t0,problem_type)

% use lasso to select features among polynomial predictors (degree = 3)

% depending on the probelm type, the tuned lambda is different
lam = [10,100,10];
X_train = [ones(length(train_data_sampled(:,1)),1),train_data_sampled(:,1),...
    train_data_sampled(:,1).^2, train_data_sampled(:,1).^3, ...
    train_data_sampled(:,2), train_data_sampled(:,2).^2,...
    train_data_sampled(:,2).^3, train_data_sampled(:,3), ...
    train_data_sampled(:,3).^2, train_data_sampled(:,3).^3, ...
    train_data_sampled(:,4), train_data_sampled(:,4).^2, ...
    train_data_sampled(:,4).^3, train_data_sampled(:,6), ...
    train_data_sampled(:,6).^2, train_data_sampled(:,6).^3, ...
    train_data_sampled(:,7), train_data_sampled(:,7).^2, ...
    train_data_sampled(:,7).^3];
Y_train = train_data_sampled(:,5);
dim = size(X_train,2);
lambda = lam(problem_type); % chosen by cross validation
[beta, FitInfo] = lasso(X_train(:,2:dim), Y_train, 'Lambda',lambda);
intercept = FitInfo.Intercept;

test_data.minute=minutes(test_data.time-t0);
test_data_mat = table2array(test_data(:,[7,2,3,4,5,6]));

X_test = [ones(length(test_data_mat(:,1)),1),test_data_mat(:,1),...
    test_data_mat(:,1).^2, test_data_mat(:,1).^3, ...
    test_data_mat(:,2), test_data_mat(:,2).^2,...
    test_data_mat(:,2).^3, test_data_mat(:,3), ...
    test_data_mat(:,3).^2, test_data_mat(:,3).^3, ...
    test_data_mat(:,4), test_data_mat(:,4).^2, ...
    test_data_mat(:,4).^3, test_data_mat(:,5), ...
    test_data_mat(:,5).^2, test_data_mat(:,5).^3, ...
    test_data_mat(:,6), test_data_mat(:,6).^2, ...
    test_data_mat(:,6).^3];
X_test_CV = [ones(length(test_CV(:,1)),1),test_CV(:,1),...
    test_CV(:,1).^2, test_CV(:,1).^3, ...
    test_CV(:,2), test_CV(:,2).^2,...
    test_CV(:,2).^3, test_CV(:,3), ...
    test_CV(:,3).^2, test_CV(:,3).^3, ...
    test_CV(:,4), test_CV(:,4).^2, ...
    test_CV(:,4).^3, test_CV(:,5), ...
    test_CV(:,5).^2, test_CV(:,5).^3, ...
    test_CV(:,6), test_CV(:,6).^2, ...
    test_CV(:,6).^3];

pm2d5_pred = X_test(:,2:dim)*beta + intercept*ones(length(X_test(:,1)),1);
pm2d5_CV = X_test_CV(:,2:dim)*beta + intercept*ones(length(X_test_CV(:,1)),1);
residual_square = (pm2d5_CV-test_CV(:,5)).*(pm2d5_CV-test_CV(:,5));
aux_beta_abs = abs(beta);
cost = sum(residual_square(:)) + lambda*sum(aux_beta_abs(:));
rmse = sqrt(mean(residual_square));
RMSE = [rmse,cost];%return both RMSE and cost
end

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
