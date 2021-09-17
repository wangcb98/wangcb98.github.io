edges = 0:4:60;
figure(1)
subplot(3,1,1)
histogram(RMSE3,edges,'FaceColor',[0 0.4470 0.7410])
y=0:0.01:30;
x = mean(RMSE3)*ones(length(y),1);
hold on
plot(x,y,'--','LineWidth',2)
xlabel('RMSE (Poly Lasso)')
ylabel('Count')
legend('RMSE','Mean')
ylim([0 35])
xlim([0 60])
set(gca,'FontSize',12)

subplot(3,1,2)
histogram(RMSE4,edges,'FaceColor',[0 0.4470 0.7410])
y=0:0.01:30;
x = mean(RMSE4)*ones(length(y),1);
hold on
plot(x,y,'--','LineWidth',2)
xlabel('RMSE (Sine Ridge)')
ylabel('Count')
legend('RMSE','Mean')
ylim([0 35])
xlim([0 60])
set(gca,'FontSize',12)

subplot(3,1,3)
histogram(RMSE6,edges,'FaceColor',[0 0.4470 0.7410])
y=0:0.01:30;
x = mean(RMSE6)*ones(length(y),1);
hold on
plot(x,y,'--','LineWidth',2)
xlabel('RMSE (GPR)')
ylabel('Count')
legend('RMSE','Mean')
ylim([0 35])
xlim([0 60])
set(gca,'FontSize',12)


figure(2)
histogram(RMSE_interpo,edges,'FaceColor',[0 0.4470 0.7410])
y=0:0.01:30;
x = mean(RMSE_interpo)*ones(length(y),1);
hold on
plot(x,y,'--','LineWidth',2)
xlabel('RMSE (Ensemble Method)')
ylabel('Count')
legend('RMSE','Mean')
ylim([0 40])
xlim([0 45])
set(gca,'FontSize',16)