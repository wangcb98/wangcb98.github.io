Short_term = [6,38,24,16,3,5,4,1,2,1]; % 0:4:40
Long_term = [3,32,38,7,4,6,3,2,0,1]; % 0:4:40
Interpo = [16,35,24,9,5,3,2,0,1]; % 0:4:36
edge1 = 0:4:40;
edge2 = 0:4:40;
edge3 = 0:4:36;
figure(1)

subplot(3,1,1)
histogram('BinEdges', edge1,'BinCounts',Short_term,'FaceColor',[0 0.4470 0.7410]);
y=0:0.01:30;
x = 11.05*ones(length(y),1);
hold on
plot(x,y,'--','LineWidth',3)
xlabel('RMSE (Ensemble Method: short-term)')
ylabel('Count')
legend('RMSE','Mean')
ylim([0 40])
xlim([0 45])
set(gca,'FontSize',18)

subplot(3,1,2)
histogram('BinEdges', edge2,'BinCounts',Long_term,'FaceColor',[0 0.4470 0.7410]);
y=0:0.01:30;
x = 12.17*ones(length(y),1);
hold on
plot(x,y,'--','LineWidth',3)
xlabel('RMSE (Ensemble Method: long-term)')
ylabel('Count')
legend('RMSE','Mean')
ylim([0 40])
xlim([0 45])
set(gca,'FontSize',18)

subplot(3,1,3)
histogram('BinEdges', edge3,'BinCounts',Interpo,'FaceColor',[0 0.4470 0.7410]);
y=0:0.01:30;
x = 9.28*ones(length(y),1);
hold on
plot(x,y,'--','LineWidth',3)
xlabel('RMSE (Ensemble Method: interpolation)')
ylabel('Count')
legend('RMSE','Mean')
ylim([0 40])
xlim([0 45])
set(gca,'FontSize',18)