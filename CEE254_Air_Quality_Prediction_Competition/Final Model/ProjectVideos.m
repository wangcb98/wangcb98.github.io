%% Project Visualization Plots
load('./test_phase/train_data_long_term_0_var.mat')
load('./test_phase/test_data_long_term_0_var.mat')
load('pred_data_long_term_0_var.mat')
latlim = [min(train_data.lat),max(train_data.lat)];
lonlim = [min(train_data.lon),max(train_data.lon)];
t0 = min(train_data.time);
t0_pred = min(test_data.time);
t_pred_max = max(test_data.time);
tmax = max(max(test_data.time),max(train_data.time));

myVideo = VideoWriter('longTermVideoFile'); %open video file
myVideo.FrameRate = 15;  %can adjust this, 5 - 10 works well for me
open(myVideo)
figure(1)
gx = subplot(3,1,[1,2],geoaxes);
geolimits(gx,latlim,lonlim)
geobasemap(gx,'satellite')
hold on
minVal = 20;
maxVal = 150;
colormap('jet')
caxis([minVal,maxVal])
colorbar
title('PM 2.5 Pollution');
sx = subplot(3,1,3);
hold on
plot(sx,train_data.time,train_data.pm2d5,'kx')
plot(sx,test_data.time,pred_pm2d5,'rx');
xlabel(sx,'Time');
ylabel(sx,'PM2.5 (\mu g/m^3)');
grid on;
legend(sx,'Observed PM2.5','Predicted PM2.5');
ylim(sx,[0,300]);
% img = imread('Map1.png'); % Map1 for inter and long, Map 2 for short
% image(img,'XData',[117.6636,117.8359],'YData',[39.0464,39.232]);
%axis('equal')
[train_data_sampled] = movAvg(train_data);
if t0_pred > max(train_data.time)
    dt1 = 30;
    dtp = minutes(test_data.time(2)-test_data.time(1));
    indx1 = train_data_sampled(:,1)<minutes(t0_pred-t0);
    train1time = 0:1:max(train_data_sampled(indx1,1)); 
    testtime = minutes(t0_pred-t0):dtp:minutes(t_pred_max-t0);
    time = minutes(test_data.time-test_data.time(1));
    test_data.time = time;
    test_data = table2array(test_data);
    for i = 1:dt1:length(train1time)
        t = t0+minutes(i);
        X1 = train_data_sampled(train_data_sampled(:,1)==i,6);
        Y1 = train_data_sampled(train_data_sampled(:,1)==i,7);
        Z1 = train_data_sampled(train_data_sampled(:,1)==i,5);
        [myVideo] = plotTrain(gx,sx,X1,Y1,Z1,t,myVideo,dt1);
    end
    for i = 1:length(testtime)
        t = t0_pred+minutes(i);
        X1 = test_data(and(test_data(:,1)>=(i-1)*dtp,test_data(:,1)<i*dtp),5);
        Y1 = test_data(and(test_data(:,1)>=(i-1)*dtp,test_data(:,1)<i*dtp),6);
        Z1 = pred_pm2d5(and(test_data(:,1)>=(i-1)*dtp,test_data(:,1)<i*dtp));
        [myVideo] = plotTest(gx,sx,X1,Y1,Z1,t,myVideo,dtp);
    end
else
    dt1 = 5;
    dtp = minutes(test_data.time(2)-test_data.time(1));
    dt2 = 5;
    indx1 = train_data_sampled(:,1)<minutes(t0_pred-t0);
    indx2 = train_data_sampled(:,1)>minutes(t_pred_max-t0);
    train1time = 0:1:max(train_data_sampled(indx1,1));
    train2time = minutes(t_pred_max-t0)+dtp:1:max(train_data_sampled(indx2,1));    
    testtime = minutes(t0_pred-t0):dtp:minutes(t_pred_max-t0);
    time = minutes(test_data.time-test_data.time(1));
    test_data.time = time;
    test_data = table2array(test_data);
    for i = 1:dt1:length(train1time)
        t = t0+minutes(i);
        X1 = train_data_sampled(train_data_sampled(:,1)==i,6);
        Y1 = train_data_sampled(train_data_sampled(:,1)==i,7);
        Z1 = train_data_sampled(train_data_sampled(:,1)==i,5);
        [myVideo] = plotTrain(gx,sx,X1,Y1,Z1,t,myVideo,dt1);
    end
    for i = 1:length(testtime)
        t = t0_pred+minutes(i);
        X1 = test_data(and(test_data(:,1)>=(i-1)*dtp,test_data(:,1)<i*dtp),5);
        Y1 = test_data(and(test_data(:,1)>=(i-1)*dtp,test_data(:,1)<i*dtp),6);
        Z1 = pred_pm2d5(and(test_data(:,1)>=(i-1)*dtp,test_data(:,1)<i*dtp));
        [myVideo] = plotTest(gx,sx,X1,Y1,Z1,t,myVideo,dtp);
    end
    for i = 1:dt2:length(train2time)
        t = t_pred_max+minutes(i);
        X1 = train_data_sampled(train_data_sampled(:,1)==i,6);
        Y1 = train_data_sampled(train_data_sampled(:,1)==i,7);
        Z1 = train_data_sampled(train_data_sampled(:,1)==i,5);
        [myVideo] = plotTrain(gx,sx,X1,Y1,Z1,t,myVideo,dt2);
    end
end
% hold on
% minVal = 20;
% maxVal = 150;
% colormap('jet')
% caxis([minVal,maxVal])
% colorbar
% title('PM 2.5 Pollution in Tianjin, China')
% dayText = t0+duration(t,0,0);
% set(gca,'xtick',[])
% set(gca,'ytick',[])
% for j = 1:length(t)
%     movingPlot = geoscatter(Xval(j,:),-Yval(j,:)+2*39.1392,200,avg_PM2d5(j,:),'filled');
%     movingPlot.MarkerFaceAlpha = 0.6;
%     movingPlot.MarkerEdgeAlpha = 0.2;
%     dayLabel = text(117.68,39.06,string(datetime(dayText(j),'Format','MMMM d, yyyy h:mm a')),'Color','white');
%     pause(0.15)
%     frame = getframe(gcf); %get frame
%     writeVideo(myVideo, frame);
%     delete(movingPlot)
%     delete(dayLabel)
% end

close(myVideo);

function [train_data_sampled] = movAvg(train_data)
% Function converts the input training data from 3 second samples to one
% minute average samples. Function divides input spatial coordinates into
% defined grid and assigns each point to one of the grid squares. Time
% averages are computed at each spatial square where data exists.
gridType = 'Cartesian'; % Options: 'Cartesian', 'Polar'
gridElements = 100; % nElements along one side of cartisian grid or rings of polar grid.
%timeStep = 1; % Time in minutes to group data.
round_to = 0;
% if problem_type == 2
%     round_to = -1; % Use 10 minute intervals for the 7-day problem
% end
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
        train_time = round(minutes(train_data.time-min(train_data.time)),round_to);
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
    case 'Polar' % Option not developed.
end
end
function [myVideo] = plotTest(axes1,axes2,X1,Y1,Z1,t2,myVideo,dt)
movingPlot = geoscatter(axes1,X1,Y1,200,Z1,'filled');
movingPlot.MarkerFaceAlpha = 0.6;
movingPlot.MarkerEdgeColor = 'r';
movingLine = plot(axes2,[t2,t2],[0,500],'r-');
legend(axes2,'Observed PM2.5','Predicted PM2.5','Current Time');
%pause(0.0001*dt)
frame = getframe(gcf); %get frame
writeVideo(myVideo, frame);
delete(movingPlot)
delete(movingLine)
end
function [myVideo] = plotTrain(axes1,axes2,X1,Y1,Z1,t2,myVideo,dt)
movingPlot = geoscatter(axes1,X1,Y1,200,Z1,'filled');
movingPlot.MarkerFaceAlpha = 0.6;
movingPlot.MarkerEdgeAlpha = 0.2;
movingLine = plot(axes2,[t2,t2],[0,500],'r-');
legend(axes2,'Observed PM2.5','Predicted PM2.5','Current Time');
%pause(0.0001*dt)
frame = getframe(gcf); %get frame
writeVideo(myVideo, frame);
delete(movingPlot)
delete(movingLine)
end
