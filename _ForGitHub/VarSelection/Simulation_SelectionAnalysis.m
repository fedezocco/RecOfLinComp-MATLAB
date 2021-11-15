%% Implemented by Federico Zocco, last update: 15/11/2021

%% To analyse the variables selected by FSCA, SPBR and MPBR considering 100 simulations.
% A single simulation consist of generating a realisation of the simulated
% dataset (see code to generate it) and executing FSCA, SPBR and MPBR on
% each realisation. The sequence of variables selected by each method is
% stored.

% NOTE: this code performs a standardization of Xsimulated, whereas
% "Simulation_VarSel_RandomSets.m" just removes the mean; this is because I
% wrote this code more than one year later than the other, time during which I learnt that 
% the best practice in machine learning is to standardise as I do here; however, just removing 
% the mean does not compromise the validity of the results given by "Simulation_VarSel_RandomSets.m",
% hence neither I changed that code nor I discarded its results. This code
% has been used to generate Fig. 4 in [1].

% REFERENCES:
% [1] F. Zocco and S. McLoone, "Recovery of linear components: Reduced
% complexity autoencoder designs," https://arxiv.org/pdf/2012.07543.pdf,
% 2020.
% [2] L. Puggini and S. McLoone, "Forward selection component analysis:
% Algorithms and applications," IEEE Transactions on Pattern Analysis and
% Machine Intelligence, vol. 39, no. 12, pp. 2395-2408, 2017.
% [3] Y. Bengio, P. Lamblin, D. Popovici, and H. Larochelle, "Greedy
% layer-wise training of deep networks," in Advances in Neural Information
% Processing Systems, pp. 153-160, 2007.
% [4] D. Erhan, Y. Bengio, A. Courville, P.-A. Manzagol, P. Vincent, and S.
% Bengio, "Why does unsupervised pre-training help deep learning?," Journal
% of Machine Learning Research, vol. 11, no. Feb., pp. 625-660, 2010.


clear all;
datasetSelector = 1;
k = 10;

switch datasetSelector
    case 1
        load('...\Datasets\Xsimulated1_NoNoise_forSelectionAnalysis')
        X = Xstore;
    
    case 2
        load('...\Datasets\Xsimulated1_03Noise_forSelectionAnalysis')
        X = Xstore;
        
    case 3
        load('...\Datasets\Xsimulated1_01Noise_forSelectionAnalysis')
        X = Xstore;
        
    case 4 % to see the selection in case only the first 10 variables are considered (i.e. only X_nonLin)
        load('...\Datasets\Xsimulated1_NoNoise_forSelectionAnalysis')
        X = Xstore(:,:,1);
        
end


for i = 1:size(X,3)
    actual_X = X(:,:,i);
    if datasetSelector == 4  
        actual_X = actual_X(:,1:10);
    end
    % Standardization before training:
    v_Mean = mean(actual_X);
    v_Std = std(actual_X) + eps;
    actual_X_ZeroMean = actual_X - ones(size(actual_X,1),1)*v_Mean;
    actual_X_Standardized = actual_X_ZeroMean./(ones(size(actual_X,1),1)*v_Std);
    actual_X_ReadyForTraining = actual_X_Standardized;
    
    % Variable selection:
    [~, ~, ~, v_selected_FSCA] = fsca(actual_X_ReadyForTraining, k);
    v_selected_SPBR = spbr(actual_X_ReadyForTraining, v_selected_FSCA);
    v_selected_MPBR = mpbr(actual_X_ReadyForTraining, v_selected_FSCA);
    
    % Storage of selection sequences:
    M_selected_FSCA(i,:) = v_selected_FSCA;
    M_selected_SPBR(i,:) = v_selected_SPBR;
    M_selected_MPBR(i,:) = v_selected_MPBR;
    
end

% Below calculate the cumulative selection of the first 10
% variables at each simulation.
for i = 1:size(M_selected_FSCA,1)
    counterForVariables1to10_FSCA = 0;
    counterForVariables1to10_SPBR = 0;
    counterForVariables1to10_MPBR = 0;
    for v = 1:10
        if sum(M_selected_FSCA(i,:)==v)
            counterForVariables1to10_FSCA = counterForVariables1to10_FSCA + 1;
        end
        if sum(M_selected_SPBR(i,:)==v)
            counterForVariables1to10_SPBR = counterForVariables1to10_SPBR + 1;
        end
        if sum(M_selected_MPBR(i,:)==v)
            counterForVariables1to10_MPBR = counterForVariables1to10_MPBR + 1;
        end
    end
    v_counterForVariables1to10_FSCA(i,1) = counterForVariables1to10_FSCA;
    v_counterForVariables1to10_SPBR(i,1) = counterForVariables1to10_SPBR;
    v_counterForVariables1to10_MPBR(i,1) = counterForVariables1to10_MPBR;
end
    
        

%% Plots
% (1) Cumulative
figure(1)
windowSize = 20;
plot(1:windowSize, v_counterForVariables1to10_FSCA(1:windowSize), 'r-.o', 'LineWidth',6, 'MarkerSize',10)
hold on;
plot(1:windowSize, v_counterForVariables1to10_SPBR(1:windowSize), 'c-.o', 'LineWidth',6, 'MarkerSize',10)
hold on;
plot(1:windowSize, v_counterForVariables1to10_MPBR(1:windowSize), 'm-.o', 'LineWidth',6, 'MarkerSize',10)
hold on;
legend('FSCA','SPBR','MPBR');
ylabel('Cumulative selection of first 10 variables','FontSize',49)
xlim([1 windowSize])
xlabel('Simulation','FontSize',49)
set(gca,'FontSize',45) 
grid on;

% (2) Histograms
subplot(3,1,1)
histogram(M_selected_FSCA);
hold on;
histogram(M_selected_FSCA_01);
%ylabel('Selection frequency (%)','FontSize',49)
legend('\sigma^2 = 0.0','\sigma^2 = 0.01','Orientation','horizontal');
%xlabel('Variable index','FontSize',49)
xlim([1 50])
xticks([1 2 3 4 5 6 7 8 9 10 20 30 40 50])
xticklabels({'1','','','','','','','','','10','20','30','40','50'})
set(gca,'FontSize',30) 
title('FSCA')

subplot(3,1,2)
histogram(M_selected_SPBR);
hold on;
histogram(M_selected_SPBR_01);
ylabel('Selection frequency (%)','FontSize',49)
%legend('\sigma_n = 0.0','\sigma_n = 0.1');
%xlabel('Variable index','FontSize',49)
xlim([1 50])
xticks([1 2 3 4 5 6 7 8 9 10 20 30 40 50])
xticklabels({'1','','','','','','','','','10','20','30','40','50'})
set(gca,'FontSize',30)
title('SPBR')

subplot(3,1,3)
histogram(M_selected_MPBR);
hold on;
histogram(M_selected_MPBR_01);
%ylabel('Selection frequency (%)','FontSize',49)
%legend('\sigma_n = 0.0','\sigma_n = 0.1');
xlabel('Variable index','FontSize',49)
xlim([1 50])
xticks([1 2 3 4 5 6 7 8 9 10 20 30 40 50])
xticklabels({'1','','','','','','','','','10','20','30','40','50'})
set(gca,'FontSize',30)
title('MPBR')