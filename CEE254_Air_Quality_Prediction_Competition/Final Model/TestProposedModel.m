%test short-term prediction accuracy (round 2)
problem_type = 1;
nrealizations = 50;
RMSE2 = zeros(nrealizations,1);
for i = 1:nrealizations
    [train_data,test_data] = generateTrainingData(problem_type);
    ground_truth = test_data.pm2d5;
    test_data.pm2d5 = [];
    [pred_pm2d5] = pm2d5_pred_model(train_data, test_data, problem_type);
    residual = ground_truth - pred_pm2d5;
    RMSE2(i) = sqrt(mean(residual.*residual));
    fprintf('i = %f\n', i)
end


%test long-term prediction accuracy
problem_type = 2;
nrealizations = 100;
ground_truth = zeros(24,100);
pred_pm2d5_simplelinear=zeros(24,100);
pred_pm2d5_simplelinearlasso=zeros(24,100);
pred_pm2d5_polylasso=zeros(24,100);
pred_pm2d5_sineridge=zeros(24,100);
pred_pm2d5_sinelasso=zeros(24,100);
pred_pm2d5_gpr=zeros(24,100);
RMSE_long_term = zeros(nrealizations,1);
RMSE1=zeros(nrealizations,1);
RMSE2=zeros(nrealizations,1);
RMSE3=zeros(nrealizations,1);
RMSE4=zeros(nrealizations,1);
RMSE5=zeros(nrealizations,1);
RMSE6=zeros(nrealizations,1);
for i = 1:nrealizations
    [train_data,test_data] = generateTrainingData(problem_type);
    ground_truth(:,i) = test_data.pm2d5;
    test_data.pm2d5 = [];
    [pred_pm2d5,preds] = pm2d5_pred_model(train_data, test_data, problem_type);
    pred_pm2d5_simplelinear(:,i)=preds(:,1);
    pred_pm2d5_simplelinearlasso(:,i) = preds(:,2); 
    pred_pm2d5_polylasso(:,i) = preds(:,3);
    pred_pm2d5_sineridge(:,i)= preds(:,4);
    pred_pm2d5_sinelasso(:,i) = preds(:,5);
    pred_pm2d5_gpr(:,i) = preds(:,6);
    residual = ground_truth(:,i) - pred_pm2d5;
    RMSE_long_term(i) = sqrt(mean(residual.*residual));
    RMSE_long_term(i)
    
    Residual_1 = ground_truth(:,i) - pred_pm2d5_simplelinear(:,i);
    RMSE1(i) = sqrt(mean(Residual_1.*Residual_1));
    RMSE1(i)
    
    Residual_2 = ground_truth(:,i) - pred_pm2d5_simplelinearlasso(:,i);
    RMSE2(i) = sqrt(mean(Residual_2.*Residual_2));
    RMSE2(i)
    
    Residual_3 = ground_truth(:,i) - pred_pm2d5_polylasso(:,i);
    RMSE3(i) = sqrt(mean(Residual_3.*Residual_3));
    RMSE3(i)
    
    Residual_4 = ground_truth(:,i) - pred_pm2d5_sineridge(:,i);
    RMSE4(i) = sqrt(mean(Residual_4.*Residual_4));
    RMSE4(i)
    
    Residual_5 = ground_truth(:,i) - pred_pm2d5_sinelasso(:,i);
    RMSE5(i) = sqrt(mean(Residual_5.*Residual_5));
    RMSE5(i)
    
    Residual_6 = ground_truth(:,i) - pred_pm2d5_gpr(:,i);
    RMSE6(i) = sqrt(mean(Residual_6.*Residual_6));
    RMSE6(i)
    
    fprintf('i = %f\n', i)
end



%test interpolation accuracy (round 2)
problem_type = 3;
nrealizations = 50;
RMSE1 = zeros(nrealizations,1);
for i = 1:nrealizations
    [train_data,test_data] = generateTrainingData(problem_type);
    ground_truth = test_data.pm2d5;
    test_data.pm2d5 = [];
    [pred_pm2d5] = pm2d5_pred_model(train_data, test_data, problem_type);
    residual = ground_truth - pred_pm2d5;
    RMSE1(i) = sqrt(mean(residual.*residual));
    fprintf('i = %f\n', i)
end



% test alternative long-term prediction method
%test long-term prediction accuracy
problem_type = 2;
nrealizations = 100;
RMSE = zeros(nrealizations,1);
pred_pm2d5_gpr = zeros(24,nrealizations);
pm2d5_pred_simplelinearlasso = zeros(24,nrealizations);
pm2d5_pred_polylasso = zeros(24,nrealizations);
rmse__simplelinearlasso = zeros(nrealizations,1);
rmse_polylasso = zeros(24,nrealizations);
ground_truth = zeros(24,nrealizations);
pred_pm2d5 = zeros(24,nrealizations);
for i = 1:nrealizations
    tic
    [train_data,test_data] = generateTrainingData(problem_type);
    [train_data_sampled] = movAvg(train_data); % Time is in minutes
    ground_truth(:,i) = test_data.pm2d5;
    test_data.pm2d5 = [];
    t0 = min(train_data.time);
    TF = ismissing(train_data_sampled);
    [row,col] = find(TF);
    train_data_sample_clean = train_data_sampled(all(~isnan(train_data_sampled),2),:);
    train_interpo_x = train_data_sample_clean(:,[1,2,3,4,6,7]);
    train_interpo_y = train_data_sample_clean(:,5);
    gprMdl = fitrgp(train_interpo_x,train_interpo_y,'KernelFunction','matern32');
    t=minutes(test_data.time-t0);
    test_data.minute = t;
    [pred_pm2d5_gpr(:,i)] = predict(gprMdl,table2array(test_data(:,[7,2,3,4,5,6])));
    [pm2d5_pred_simplelinearlasso(:,i),rmse__simplelinearlasso(i)] = crossVal(@simple_linear_lasso,train_data_sampled,test_data,t0,problem_type);
    [pm2d5_pred_polylasso(:,i),rmse_polylasso(i)] = crossVal(@poly_lasso,train_data_sampled,test_data,t0,problem_type);
    pred_pm2d5(:,i) = [pred_pm2d5_gpr(:,i),pm2d5_pred_simplelinearlasso(:,i),pm2d5_pred_polylasso(:,i)]*[0.35;0.3;0.35];
    Residual = pred_pm2d5(:,i)-ground_truth(:,i);
    RMSE(i) = sqrt(mean(Residual.*Residual));
    toc
    fprintf('i = %f\n', i)
end

RMSE_simplelinearlasso = zeros(20,1);
for i = 1:20
    Residual = pm2d5_pred_simplelinearlasso(:,i)-ground_truth(:,i);
    RMSE_simplelinearlasso(i) = sqrt(mean(Residual.*Residual));
end 
mean(RMSE_simplelinearlasso(~isnan(RMSE_simplelinearlasso)))

RMSE_polylasso = zeros(20,1);
for i = 1:20
    Residual = pm2d5_pred_polylasso(:,i)-ground_truth(:,i);
    RMSE_polylasso(i) = sqrt(mean(Residual.*Residual));
end 
mean(RMSE_polylasso(~isnan(RMSE_polylasso)))

RMSE_gpr = zeros(20,1);
for i = 1:20
    Residual = pred_pm2d5_gpr(:,i)-ground_truth(:,i);
    RMSE_gpr(i) = sqrt(mean(Residual.*Residual));
end 
mean(RMSE_gpr(~isnan(RMSE_gpr)))




%test short-term prediction accuracy (Ben's lambdas for lasso techniques)
problem_type = 1;
nrealizations = 20;
RMSE4 = zeros(nrealizations,1);
for i = 1:nrealizations
    [train_data,test_data] = generateTrainingData(problem_type);
    ground_truth = test_data.pm2d5;
    test_data.pm2d5 = [];
    [pred_pm2d5] = pm2d5_pred_model(train_data, test_data, problem_type);
    residual = ground_truth - pred_pm2d5;
    RMSE4(i) = sqrt(mean(residual.*residual));
    RMSE4(i)
    fprintf('i = %f\n', i)
end


%test short-term prediction accuracy
problem_type = 1;
nrealizations = 100;
ground_truth = zeros(3,100);
pred_pm2d5_simplelinear=zeros(3,100);
pred_pm2d5_simplelinearlasso=zeros(3,100);
pred_pm2d5_polylasso=zeros(3,100);
pred_pm2d5_sineridge=zeros(3,100);
pred_pm2d5_sinelasso=zeros(3,100);
pred_pm2d5_gpr=zeros(3,100);
RMSE_short_term = zeros(nrealizations,1);
RMSE1=zeros(nrealizations,1);
RMSE2=zeros(nrealizations,1);
RMSE3=zeros(nrealizations,1);
RMSE4=zeros(nrealizations,1);
RMSE5=zeros(nrealizations,1);
RMSE6=zeros(nrealizations,1);
for i = 1:nrealizations
    [train_data,test_data] = generateTrainingData(problem_type);
    ground_truth(:,i) = test_data.pm2d5;
    test_data.pm2d5 = [];
    [pred_pm2d5,preds] = pm2d5_pred_model(train_data, test_data, problem_type);
    pred_pm2d5_simplelinear(:,i)=preds(:,1);
    pred_pm2d5_simplelinearlasso(:,i) = preds(:,2); 
    pred_pm2d5_polylasso(:,i) = preds(:,3);
    pred_pm2d5_sineridge(:,i)= preds(:,4);
    pred_pm2d5_sinelasso(:,i) = preds(:,5);
    pred_pm2d5_gpr(:,i) = preds(:,6);
    residual = ground_truth(:,i) - pred_pm2d5;
    RMSE_short_term(i) = sqrt(mean(residual.*residual));
    RMSE_short_term(i)
    
    Residual_1 = ground_truth(:,i) - pred_pm2d5_simplelinear(:,i);
    RMSE1(i) = sqrt(mean(Residual_1.*Residual_1));
    RMSE1(i)
    
    Residual_2 = ground_truth(:,i) - pred_pm2d5_simplelinearlasso(:,i);
    RMSE2(i) = sqrt(mean(Residual_2.*Residual_2));
    RMSE2(i)
    
    Residual_3 = ground_truth(:,i) - pred_pm2d5_polylasso(:,i);
    RMSE3(i) = sqrt(mean(Residual_3.*Residual_3));
    RMSE3(i)
    
    Residual_4 = ground_truth(:,i) - pred_pm2d5_sineridge(:,i);
    RMSE4(i) = sqrt(mean(Residual_4.*Residual_4));
    RMSE4(i)
    
    Residual_5 = ground_truth(:,i) - pred_pm2d5_sinelasso(:,i);
    RMSE5(i) = sqrt(mean(Residual_5.*Residual_5));
    RMSE5(i)
    
    Residual_6 = ground_truth(:,i) - pred_pm2d5_gpr(:,i);
    RMSE6(i) = sqrt(mean(Residual_6.*Residual_6));
    RMSE6(i)
    
    fprintf('i = %f\n', i)
end


%test interpolation prediction accuracy
problem_type = 3;
nrealizations = 100;
ground_truth = zeros(12,100);
pred_pm2d5_polylasso=zeros(12,100);
pred_pm2d5_sineridge=zeros(12,100);
pred_pm2d5_gpr=zeros(12,100);
RMSE_short_term = zeros(nrealizations,1);
RMSE3=zeros(nrealizations,1);
RMSE4=zeros(nrealizations,1);
RMSE6=zeros(nrealizations,1);
for i = 1:nrealizations
    [train_data,test_data] = generateTrainingData(problem_type);
    ground_truth(:,i) = test_data.pm2d5;
    test_data.pm2d5 = [];
    [pred_pm2d5,preds] = pm2d5_pred_model(train_data, test_data, problem_type);

    pred_pm2d5_polylasso(:,i) = preds(:,1);
    pred_pm2d5_sineridge(:,i)= preds(:,2);
    pred_pm2d5_gpr(:,i) = preds(:,3);
    residual = ground_truth(:,i) - pred_pm2d5;
    RMSE_short_term(i) = sqrt(mean(residual.*residual));
    RMSE_short_term(i)
   
    
    Residual_3 = ground_truth(:,i) - pred_pm2d5_polylasso(:,i);
    RMSE3(i) = sqrt(mean(Residual_3.*Residual_3));
    RMSE3(i)
    
    Residual_4 = ground_truth(:,i) - pred_pm2d5_sineridge(:,i);
    RMSE4(i) = sqrt(mean(Residual_4.*Residual_4));
    RMSE4(i)
    
    Residual_6 = ground_truth(:,i) - pred_pm2d5_gpr(:,i);
    RMSE6(i) = sqrt(mean(Residual_6.*Residual_6));
    RMSE6(i)
    
    fprintf('i = %f\n', i)
end
