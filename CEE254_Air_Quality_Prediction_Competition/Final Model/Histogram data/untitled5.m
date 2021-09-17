RMSE3 = [10,22,17,13,8,7,6,4,2,1,1,2,0,2,0]; % 0:4:60
RMSE4 = [13,23,31,15,6,3,3,1,0,1,0,0,0,0,0]; % 0:4:60
RMSE6 = [13,30,29,9,2,3,1,0,1,0,2,2,0,0,0]; % 0:4:60
edge1 = 0:4:60;
edge2 = 0:4:60;
edge3 = 0:4:60;
figure(1)

subplot(3,1,1)
histogram('BinEdges', edge1,'BinCounts',RMSE3,'FaceColor',[0 0.4470 0.7410]);
y=0:0.01:30;
x = 15.78*ones(length(y),1);
hold on
plot(x,y,'--','LineWidth',3)
xlabel('RMSE (Poly LASSO)')
ylabel('Count')
legend('RMSE','Mean')
ylim([0 35])
xlim([0 60])
set(gca,'FontSize',18)

subplot(3,1,2)
histogram('BinEdges', edge2,'BinCounts',RMSE4,'FaceColor',[0 0.4470 0.7410]);
y=0:0.01:30;
x = 10.62*ones(length(y),1);
hold on
plot(x,y,'--','LineWidth',3)
xlabel('RMSE (Sine Ridge)')
ylabel('Count')
legend('RMSE','Mean')
ylim([0 35])
xlim([0 60])
set(gca,'FontSize',18)

subplot(3,1,3)
histogram('BinEdges', edge3,'BinCounts',RMSE6,'FaceColor',[0 0.4470 0.7410]);
y=0:0.01:30;
x = 13.32*ones(length(y),1);
hold on
plot(x,y,'--','LineWidth',3)
xlabel('RMSE (GPR)')
ylabel('Count')
legend('RMSE','Mean')
ylim([0 35])
xlim([0 60])
set(gca,'FontSize',18)