function [train_data,test_data] = generateTrainingData(problem_type)
staticTestLoc = randi(5);
switch problem_type
    case 1
        load('short_term_foshan_train_val.mat');
        t_start = rand()*4;
        t_end = t_start + 3;
        train_data = splitData([data_static(1,1:staticTestLoc-1),data_static(1,staticTestLoc+1:end)],...
            data_mobile,t_start,t_end);
        t = (1:3)/24+t_end;
        [test_data] = computeTestData(data_static(1,staticTestLoc),t);
    case 2
        load('long_term_tianjin_train_val.mat');
        t_start = rand()*20;
        t_end = t_start + 7;
        train_data = splitData([data_static(1,1:staticTestLoc-1),data_static(1,staticTestLoc+1:end)],...
            data_mobile,t_start,t_end);
        t = (1:24)/24+t_end;
        [test_data] = computeTestData(data_static(1,staticTestLoc),t);
    case 3
        load('long_term_tianjin_train_val.mat');
        t_start = rand()*21;
        t_end = t_start + 3;
        train_data = splitData([data_static(1,1:staticTestLoc-1),data_static(1,staticTestLoc+1:end)],...
            data_mobile,t_start,t_end);
        t = (1:12)/288+t_start+1.5;
        [test_data] = computeTestData(data_static(1,staticTestLoc),t);
end
end
function [train_data] = splitData(data_static,data_mobile,t_start,t_end)
dsl = length(data_static);
dml = length(data_mobile);
M = data_static{1,1};
time = days(M.time-M.time(1));
M = M(and(time>=t_start,time<t_end),:);
train_data = M;
for i = 2:dsl
    M = data_static{1,i};
    time = days(M.time-M.time(1));
    M = M(and(time>=t_start,time<t_end),:);
    train_data = [train_data;M];
end
for i = dsl+1:dml+dsl
    M = data_mobile{1,i-dsl};
    time = days(M.time-M.time(1));
    M = M(and(time>=t_start,time<t_end),:);
    train_data = [train_data;M];
end
end
function [test_data] = computeTestData(data_static,t)

M = data_static{1,1};
time = days(M.time-M.time(1));
test_data = M(1:length(t),:);
test_data.time = M.time(1) + days(t');
t = [0,t]+(t(2)-t(1))/2;
for i = 2:length(t)
    test_data{i-1,2:end} = mean(table2array(M(and(time>=t(i-1),time<t(i)),2:end)));
end
end
