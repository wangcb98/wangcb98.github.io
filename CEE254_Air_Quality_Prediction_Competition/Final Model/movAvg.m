%% movAvg
function [train_data_sampled] = movAvg(train_data)
% Function converts the input training data from 3 second samples to one
% minute average samples. Function divides input spatial coordinates into
% defined grid and assigns each point to one of the grid squares. Time
% averages are computed at each spatial square where data exists.
gridType = 'Cartesian'; % Options: 'Cartesian', 'Polar'
gridElements = 10; % nElements along one side of cartisian grid or rings of polar grid.
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
