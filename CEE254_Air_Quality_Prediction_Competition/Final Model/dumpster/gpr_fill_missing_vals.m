%% gpr_fill_missing_vals
function [gprMdl_full,train_data_sampled,rmse] = gpr_fill_missing_vals(train_data_sampled)

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

gprMdl_full = fitrgp(train_interpo_x,train_interpo_y,'KernelFunction','matern32');
for i=1:length(row)
    test_interpo_sampled = train_data_sampled(row(i),:);
    test_interpo_x = test_interpo_sampled(:,[1,2,3,4,6,7]);
    train_data_sampled(row(i),col(i))=predict(gprMdl_full,test_interpo_x);
end

TF_new = ismissing(train_data_sampled);
if sum(TF_new(:)) ~= 0
    fprintf("Something went wrong! Missing values still persist.")
    % if this happens, there probably are missing weather/time data
end
rmse = mean(err);%return the mean rmse for 5-fold Cross Validation of using this GPR
end