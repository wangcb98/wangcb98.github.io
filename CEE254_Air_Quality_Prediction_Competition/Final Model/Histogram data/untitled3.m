edges = 0:4:80;
figure(1)

subplot(3,2,1)
histogram(RMSE1,edges,'FaceColor',[0 0.4470 0.7410])
y=0:0.01:30;
x = mean(RMSE1)*ones(length(y),1);
hold on
plot(x,y,'--','LineWidth',3)
xlabel('RMSE (Simple Linear)')
ylabel('Count')
legend('RMSE','Mean')
ylim([0 35])
xlim([0 80])
set(gca,'FontSize',18)

subplot(3,2,2)
histogram(RMSE2,edges,'FaceColor',[0 0.4470 0.7410])
y=0:0.01:30;
x = mean(RMSE2)*ones(length(y),1);
hold on
plot(x,y,'--','LineWidth',3)
xlabel('RMSE (Simple Linear LASSO)')
ylabel('Count')
legend('RMSE','Mean')
ylim([0 35])
xlim([0 80])
set(gca,'FontSize',18)

subplot(3,2,3)
histogram(RMSE3,edges,'FaceColor',[0 0.4470 0.7410])
y=0:0.01:30;
x = mean(RMSE3)*ones(length(y),1);
hold on
plot(x,y,'--','LineWidth',3)
xlabel('RMSE (Poly LASSO)')
ylabel('Count')
legend('RMSE','Mean')
ylim([0 35])
xlim([0 80])
set(gca,'FontSize',18)

subplot(3,2,4)
histogram(RMSE4,edges,'FaceColor',[0 0.4470 0.7410])
y=0:0.01:30;
x = mean(RMSE4)*ones(length(y),1);
hold on
plot(x,y,'--','LineWidth',3)
xlabel('RMSE (Sine Ridge)')
ylabel('Count')
legend('RMSE','Mean')
ylim([0 35])
xlim([0 80])
set(gca,'FontSize',18)

subplot(3,2,5)
histogram(RMSE5,edges,'FaceColor',[0 0.4470 0.7410])
y=0:0.01:30;
x = mean(RMSE5)*ones(length(y),1);
hold on
plot(x,y,'--','LineWidth',3)
xlabel('RMSE (Sine LASSO)')
ylabel('Count')
legend('RMSE','Mean')
ylim([0 35])
xlim([0 80])
set(gca,'FontSize',18)

subplot(3,2,6)
histogram(RMSE6,edges,'FaceColor',[0 0.4470 0.7410])
y=0:0.01:30;
x = mean(RMSE6)*ones(length(y),1);
hold on
plot(x,y,'--','LineWidth',3)
xlabel('RMSE (GPR)')
ylabel('Count')
legend('RMSE','Mean')
ylim([0 35])
xlim([0 80])
set(gca,'FontSize',18)
