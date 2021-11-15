%% Implemented by Federico Zocco, last update: 15/11/2021

%% The script performs Monte Carlo simulations on the dataset selected by the 'switch'.
% Three methods are compared: (1) PCA, (2) PCA-SAE and (3) PCA-RLC.
% As in the name 'Simulation_DimRed_fixedSets_BandC', this
% script considers dimensionality reduction and fixed sets, i.e. the
% training and test sets are defined at the beginning and then kept fixed over
% all the simulations.

% NOTE: this code has been used to generate Table 8 and Figures 7.b, 7.c, 8 and 9 in [1], 
% which consider Scenario B and Scenario C.

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
% [5] L. van der Maaten, E. Postma, and J. van den Herik, "Dimensionality
% reduction: A comparative review," TiCC TR, vol. 005, no.1, pp. 1-35, 2009.


%===================== PARAMETERS TO SET: ======================
clear all;
datasetSelector = 1; 

switch datasetSelector
     case 1
        FractionOfDataForTraining = 0.7;
        
        load('...\Datasets\Xsimulated1_m=500noise0.1.mat')
        TrainIdx = randperm(size(Xsimulated,1),round(size(Xsimulated,1)*FractionOfDataForTraining));
        TestIdx = 1:size(Xsimulated,1);
        TestIdx(TrainIdx) = [];
        Xtrain = Xsimulated(TrainIdx,:);
        Xtest = Xsimulated(TestIdx,:);
        v_trainMean = mean(Xtrain); % Stores the mean of the process
        XtrainZeroMean = Xtrain - v_trainMean;
        XtestZeroMean = Xtest - v_trainMean; % Removes the mean defined on the TRAINING set
        XtrainReadyForTraining = XtrainZeroMean;
        XtestReadyForTest = XtestZeroMean;
    

     case 2
        FractionOfDataForTraining = 0.5;
        
        load('...\Datasets\XgasBatch3.mat')
        TrainIdx = randperm(size(XgasBatch3,1),round(size(XgasBatch3,1)*FractionOfDataForTraining));
        TestIdx = 1:size(XgasBatch3,1);
        TestIdx(TrainIdx) = [];
        Xtrain = XgasBatch3(TrainIdx,:);
        Xtest = XgasBatch3(TestIdx,:);
        v_trainMean = mean(Xtrain); % Stores the mean of the process
        v_trainStd = std(Xtrain) + eps; % Stores the standard deviation of the process
        XtrainZeroMean = Xtrain - ones(size(Xtrain,1),1)*v_trainMean;
        XtrainStandardized = XtrainZeroMean./(ones(size(Xtrain,1),1)*v_trainStd);
        XtrainReadyForTraining = XtrainStandardized;
        XtestZeroMean = Xtest - ones(size(Xtest,1),1)*v_trainMean;
        XtestStandardized = XtestZeroMean./(ones(size(Xtest,1),1)*v_trainStd);
        XtestReadyForTest = XtestStandardized;
    

     case 3
        FractionOfDataForTraining = 0.15;
        
        load('...\Datasets\USPS.mat')
        TrainIdx = randperm(size(X,1),round(size(X,1)*FractionOfDataForTraining));
        TestIdx = 1:size(X,1);
        TestIdx(TrainIdx) = [];
        Xtrain = X(TrainIdx,:);
        Xtest = X(TestIdx,:);
        v_trainMean = mean(Xtrain); % Stores the mean of the process
        XtrainZeroMean = Xtrain - v_trainMean;
        XtestZeroMean = Xtest - v_trainMean; % Removes the mean defined on the TRAINING set
        XtrainReadyForTraining = XtrainZeroMean;
        XtestReadyForTest = XtestZeroMean;
        

     case 4
        FractionOfDataForTraining = 0.7;
        
        load('...\Datasets\COIL20.mat')
        TrainIdx = randperm(size(X,1),round(size(X,1)*FractionOfDataForTraining));
        TestIdx = 1:size(X,1);
        TestIdx(TrainIdx) = [];
        Xtrain = X(TrainIdx,:);
        Xtest = X(TestIdx,:);
        v_trainMean = mean(Xtrain); % Stores the mean of the process
        XtrainZeroMean = Xtrain - v_trainMean;
        XtestZeroMean = Xtest - v_trainMean; % Removes the mean defined on the TRAINING set
        XtrainReadyForTraining = XtrainZeroMean;
        XtestReadyForTest = XtestZeroMean;
        
     
     case 5
        FractionOfDataForTraining = 0.7;
        
        load('...\Datasets\MNIST.mat')
        TrainIdx = randperm(size(X,1),round(size(X,1)*FractionOfDataForTraining));
        TestIdx = 1:size(X,1);
        TestIdx(TrainIdx) = [];
        Xtrain = X(TrainIdx,:);
        Xtest = X(TestIdx,:);
        v_trainMean = mean(Xtrain); % Stores the mean of the process
        XtrainZeroMean = Xtrain - v_trainMean;
        XtestZeroMean = Xtest - v_trainMean; % Removes the mean defined on the TRAINING set
        XtrainReadyForTraining = XtrainZeroMean;
        XtestReadyForTest = XtestZeroMean;
end

valuesOfk = [2 3 4];
h1 = [];
numOfTrainings = 70;
NumOfEpoch_PCAsae = [];
VE_LinearityThreshold = 99;

%==========================================================================





%% TRAINING phase:=========================================================
  
[k_linTrain, Ptrain_lin, Ttrain_lin, ~, v_compTime_PCA] = define_k_lin(XtrainReadyForTraining, VE_LinearityThreshold, 'PCA'); % (a) Ptrain_lin
 
for k = valuesOfk
    for j = 1:numOfTrainings
        % (1) PCA (Models to define in training: (a) Ptrain):
        Ptrain = pca_nipals(XtrainReadyForTraining,k); % PCA wants null mean.
        
        % (2) PCAsae (Models to define in training: (a) Ptrain_lin (b) SAEtrain (c) BetaTrain_PCAsae):
        v_NumOfNeuronsPerEncoder_PCAsae = [k]; % = [k] equivalence condition
        
        tic;
        v_NumOfNeuronsPerHiddenDecoder_PCAsae = flip(v_NumOfNeuronsPerEncoder_PCAsae(1,1:length(v_NumOfNeuronsPerEncoder_PCAsae)-1));
        SAEtrain = feedforwardnet([v_NumOfNeuronsPerEncoder_PCAsae  v_NumOfNeuronsPerHiddenDecoder_PCAsae]);
        if ~isempty(NumOfEpoch_PCAsae)
           SAEtrain.trainParam.epochs = NumOfEpoch_PCAsae; 
        end
        [SAEtrain, trainState_SAE] = train(SAEtrain, Ttrain_lin', Ttrain_lin'); % (b) SAEtrain
        Ttrain_lin_hat = SAEtrain(Ttrain_lin')';
        BetaTrain_PCAsae = pinv(Ttrain_lin_hat)*XtrainReadyForTraining; % (c) BetaTrain_PCAsae
        M_compTime_PCAsae(j,find(valuesOfk==k)) = toc + v_compTime_PCA(1,k_linTrain);
        M_EpochsNum_PCAsae(j,find(valuesOfk==k)) = trainState_SAE.epoch(1,length(trainState_SAE.epoch));
        M_TimePerEpoch_PCAsae{j,:} = trainState_SAE.time;
        
        % (3) RLC (Models to define in training: (a) Ptrain (b) RLCtrain (c) BetaTrain_RLC):
        v_HiddenLayers_RLC = [2*k]; % = [2*k] equivalence condition
        
        tic;
        % (a) Ptrain (currently taken from PCA above)
        [RLCtrain, Ttrain_discarded_hat, trainState_RLC] = rlc_pca(Ttrain_lin, k_linTrain, k, v_HiddenLayers_RLC); % (b) RLCtrain
        BetaTrain_RLC = pinv([Ttrain_lin(:,1:k)  Ttrain_discarded_hat])*XtrainReadyForTraining; % (c) BetaTrain_RLC
        M_compTime_RLC(j,find(valuesOfk==k)) = toc + v_compTime_PCA(1,k_linTrain);
        M_EpochsNum_RLC(j,find(valuesOfk==k)) = trainState_RLC.epoch(1,length(trainState_RLC.epoch));
        M_TimePerEpoch_RLC{j,:} = trainState_RLC.time;
        %==========================================================================
        
        
        
        
        
        %% TEST phase:=============================================================
        % (1) PCA:
        Ttest = XtestReadyForTest*Ptrain;
        XtestReadyForTest_hat = Ttest*Ptrain';
        VE_Test = VarExp(XtestReadyForTest, XtestReadyForTest_hat);
        M_VEtest_PCA(j,find(valuesOfk==k)) = VE_Test;
        
        % (2) PCAsae:
        Ttest_lin = XtestReadyForTest*Ptrain_lin; % Uses (a)
        Ttest_lin_hat = SAEtrain(Ttest_lin')'; % Uses (b)
        XtestReadyForTest_hat = Ttest_lin_hat*BetaTrain_PCAsae; % Uses (c)
        VE_Test = VarExp(XtestReadyForTest, XtestReadyForTest_hat);
        M_VEtest_PCAsae(j,find(valuesOfk==k)) = VE_Test;
        
        % (3) RLC:
        Ttest = XtestReadyForTest*Ptrain; % Uses (a)
        Ttest_discarded_hat = RLCtrain(Ttest')'; % Uses (b)
        XtestReadyForTest_hat = [Ttest  Ttest_discarded_hat]*BetaTrain_RLC; % Uses (c)
        VE_Test = VarExp(XtestReadyForTest, XtestReadyForTest_hat);
        M_VEtest_RLC(j,find(valuesOfk==k)) = VE_Test;
        %==========================================================================
        
        
        
        
        %% Just to check the code, on training set:
        % (2) PCAsae:
        Ttrain_lin = XtrainReadyForTraining*Ptrain_lin; % Uses (a)
        Ttrain_lin_hat = SAEtrain(Ttrain_lin')'; % Uses (b)
        XtrainReadyForTraining_hat = Ttrain_lin_hat*BetaTrain_PCAsae; % Uses (c)
        VE_train = VarExp(XtrainReadyForTraining, XtrainReadyForTraining_hat);
        M_VEtrain_PCAsae(j,find(valuesOfk==k)) = VE_train;
        % (3) RLC:
        Ttrain = XtrainReadyForTraining*Ptrain; % Uses (a)
        Ttrain_discarded_hat = RLCtrain(Ttrain')'; % Uses (b)
        XtrainReadyForTraining_hat = [Ttrain  Ttrain_discarded_hat]*BetaTrain_RLC; % Uses (c)
        VE_train = VarExp(XtrainReadyForTraining, XtrainReadyForTraining_hat);
        M_VEtrain_RLC(j,find(valuesOfk==k)) = VE_train;
    end
end
    


%%%--------------------------------- PLOTS:
%% VE:
figure;
errorbar(valuesOfk, mean(M_VEtest_PCA),std(M_VEtest_PCA)./2, std(M_VEtest_PCA)./2,'r-o','LineWidth',4,'MarkerSize',10,'CapSize',15)
hold on;
errorbar(valuesOfk, mean(M_VEtest_PCAsae), std(M_VEtest_PCAsae)./2, std(M_VEtest_PCAsae)./2,'b-o','LineWidth',4,'MarkerSize',10,'CapSize',15)
hold on;
errorbar(valuesOfk, mean(M_VEtest_RLC), std(M_VEtest_RLC)./2, std(M_VEtest_RLC)./2,'k-.o','LineWidth',4,'MarkerSize',10,'CapSize',15)
hold on;
errorbar(valuesOfk, mean(M_VEtrain_PCAsae), std(M_VEtrain_PCAsae)./2, std(M_VEtrain_PCAsae)./2,'c--o','LineWidth',4,'MarkerSize',10,'CapSize',15)
hold on;
errorbar(valuesOfk, mean(M_VEtrain_RLC), std(M_VEtrain_RLC)./2, std(M_VEtrain_RLC)./2, 'm-.o','LineWidth',4,'MarkerSize',10,'CapSize',15)
hold on;
legend('PCA(OnTest)','PCAsae(OnTest)','RLC(OnTest)','PCAsae(OnTraining)','RLC(OnTraining)');
ylabel('VE (%)','FontSize',30)
xlabel('Values of k','FontSize',30)
set(gca,'FontSize',30) 


%% Epochs vs. Simulations:
figure;
colororder({'b','k'})
yyaxis left
plot(1:numOfTrainings, M_EpochsNum_PCAsae(:,1),'b-o','LineWidth',6,'MarkerSize',10)
ylabel('Epochs required (SAE)','FontSize',50)
yyaxis right
plot(1:numOfTrainings, M_EpochsNum_RLC(:,1),'k-o','LineWidth',6,'MarkerSize',10)
ylabel('Epochs required (RLC)','FontSize',50)
legend('PCA-SAE','PCA-RLC');
xlabel('Simulations','FontSize',50)
xlim([1 numOfTrainings])
set(gca,'FontSize',50)


%% Time vs. Epochs:
NumEpochToPlot = 10; 
for i = 1:size(M_TimePerEpoch_RLC,1)
   M_TimePerEpoch_InitialEpochs_PCAsae(i,:) = M_TimePerEpoch_PCAsae{i,1}(1,1:NumEpochToPlot);
   M_TimePerEpoch_InitialEpochs_RLC(i,:) = M_TimePerEpoch_RLC{i,1}(1,1:NumEpochToPlot);
end
% 
% figure;
% plot(1:NumEpochToPlot, mean(M_TimePerEpoch_InitialEpochs_PCAsae),'b-o','LineWidth',4,'MarkerSize',10)
% hold on;
% plot(1:NumEpochToPlot, mean(M_TimePerEpoch_InitialEpochs_RLC),'k-o','LineWidth',4,'MarkerSize',10)
% hold on;
% legend('PCAsae','RLC');
% ylabel('Mean time (s)','FontSize',30)
% xlabel('Epochs','FontSize',30)
% set(gca,'FontSize',30)