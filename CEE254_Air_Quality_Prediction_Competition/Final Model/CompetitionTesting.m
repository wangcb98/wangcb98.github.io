%% Competition Testing
% 
% CEE 254: Data Analytics 
% Chenbo Wang
% Ben Flanagan
% Fall 2020
%
%
%
runCase1 = false;
runCase2 = true;
runCase3 = false;

var = 0; % Can be 0, 5, or 10 for different cases



%% Case 1: Short Term Prediction
if runCase1
%
%
train_location = ['./test_phase/train_data_short_term_',num2str(var),'_var.mat'];
test_location = ['./test_phase/test_data_short_term_',num2str(var),'_var.mat'];
train_data = load(train_location);
test_data = load(test_location);
train_data = train_data.train_data;
test_data = test_data.test_data;
figure
[pred_pm2d5] = pm2d5_pred_model(train_data, test_data, 1);
plot(train_data.time,train_data.pm2d5,'k.')
hold on
%plot(test_data.time,test_data.pm2d5,'ro')
plot(test_data.time,pred_pm2d5,'rx')
hold off
%rmse = sqrt(sum((test_data.pm2d5-pred_pm2d5).^2)/length(pred_pm2d5));
%title(['RMSE = ',num2str(rmse)]);
save(['short_term_prediction','_',num2str(var),'.mat'],'pred_pm2d5');

end
%% Case 2: Long Term Prediction
if runCase2
%
%
%
train_location = ['./test_phase/train_data_long_term_',num2str(var),'_var.mat'];
test_location = ['./test_phase/test_data_long_term_',num2str(var),'_var.mat'];
train_data = load(train_location);
test_data = load(test_location);
train_data = train_data.train_data;
test_data = test_data.test_data;
figure
[pred_pm2d5] = pm2d5_pred_model(train_data, test_data, 2);
plot(train_data.time,train_data.pm2d5,'k.')
hold on
%plot(test_data.time,test_data.pm2d5,'ro')
plot(test_data.time,pred_pm2d5,'rx')
hold off
%rmse = sqrt(sum((test_data.pm2d5-pred_pm2d5).^2)/length(pred_pm2d5));
%title(['RMSE = ',num2str(rmse)]);

%the plot may be more interpretable if we seperate for different locations?

end
%% Case 3: Interpolation Problem
if runCase3
%
%
%
train_location = ['./test_phase/train_data_interpolation_',num2str(var),'_var.mat'];
test_location = ['./test_phase/test_data_interpolation_',num2str(var),'_var.mat'];
train_data = load(train_location);
test_data = load(test_location);
train_data = train_data.train_data;
test_data = test_data.test_data;
figure
[pred_pm2d5] = pm2d5_pred_model(train_data, test_data, 3);
plot(train_data.time,train_data.pm2d5,'k.')
hold on
%plot(test_data.time,test_data.pm2d5,'ro')
plot(test_data.time,pred_pm2d5,'rx')
hold off
save(['long_term_prediction','_',num2str(var),'.mat'],'pred_pm2d5');
% rmse = sqrt(sum((test_data.pm2d5-pred_pm2d5).^2)/length(pred_pm2d5));
% title(['RMSE = ',num2str(rmse)]);
% disp(['rmse = ',num2str(rmse)])
end
