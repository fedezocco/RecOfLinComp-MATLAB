%% Implemented by Federico Zocco, last update: 15/11/2021

%% The script performs Monte Carlo simulations on the dataset selected by the 'switch'.
% Three methods are compared: (1) PCA, (2) PCA-SAE and (3) PCA-RLC. 
% As in the name 'Simulation_DimRed_randomSets', this
% script considers dimensionality reduction and random sets, i.e. the
% training and test sets are randomly redefined at the beginning of each simulation.

% NOTE: this code has been used to generate figure 7.a in [1], which considers Scenario A.

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
        NumOfSimulations = 70;
        FractionOfDataForTraining = 0.7;
        h1 = 4;
        valuesOfk = [1 2 3 4 5 6 7];
        NumOfEpoch_PCAsae = [];
        VE_LinearityThreshold = 99;
        
        load('...\Datasets\Xsimulated1_m=500noise0.1.mat')
        for i = 1:NumOfSimulations
            TrainIdx = randperm(size(Xsimulated,1),round(size(Xsimulated,1)*FractionOfDataForTraining));
            TestIdx = 1:size(Xsimulated,1);
            TestIdx(TrainIdx) = [];
            Xtrain = Xsimulated(TrainIdx,:);
            Xtest = Xsimulated(TestIdx,:);
            v_trainMean = mean(Xtrain); % Stores the mean of the process
            XtrainZeroMean = Xtrain - v_trainMean;
            XtestZeroMean = Xtest - v_trainMean; % Removes the mean defined on the TRAINING set
            XtrainReadyForTraining(:,:,i) = XtrainZeroMean;
            XtestReadyForTest(:,:,i) = XtestZeroMean;
        end

        
     case 2
        NumOfSimulations = 1;
        FractionOfDataForTraining = 0.7;
        h1 = 4;
        valuesOfk = [1 2 3 4 5 6 7];
        NumOfEpoch_PCAsae = [];
        VE_LinearityThreshold = 98;
        
        load('...\Datasets\XgasBatch3.mat')
        for i = 1:NumOfSimulations
            TrainIdx = randperm(size(XgasBatch3,1),round(size(XgasBatch3,1)*FractionOfDataForTraining));
            TestIdx = 1:size(XgasBatch3,1);
            TestIdx(TrainIdx) = [];
            Xtrain = XgasBatch3(TrainIdx,:);
            Xtest = XgasBatch3(TestIdx,:);
            v_trainMean = mean(Xtrain); % Stores the mean of the process
            v_trainStd = std(Xtrain) + eps; % Stores the standard deviation of the process
            XtrainZeroMean = Xtrain - ones(size(Xtrain,1),1)*v_trainMean;
            XtrainStandardized = XtrainZeroMean./(ones(size(Xtrain,1),1)*v_trainStd);
            XtrainReadyForTraining(:,:,i) = XtrainStandardized;
            XtestZeroMean = Xtest - ones(size(Xtest,1),1)*v_trainMean;
            XtestStandardized = XtestZeroMean./(ones(size(Xtest,1),1)*v_trainStd);
            XtestReadyForTest(:,:,i) = XtestStandardized;
        end
        
     
     case 3
        NumOfSimulations = 1;
        FractionOfDataForTraining = 0.15;
        h1 = 3;
        valuesOfk = [4]; 
        NumOfEpoch_PCAsae = 25;
        VE_LinearityThreshold = 99;
        
        load('...\Datasets\USPS.mat')
        for i = 1:NumOfSimulations
            TrainIdx = randperm(size(X,1),round(size(X,1)*FractionOfDataForTraining));
            TestIdx = 1:size(X,1);
            TestIdx(TrainIdx) = [];
            Xtrain = X(TrainIdx,:);
            Xtest = X(TestIdx,:);
            v_trainMean = mean(Xtrain); % Stores the mean of the process
            XtrainZeroMean = Xtrain - v_trainMean;
            XtestZeroMean = Xtest - v_trainMean; % Removes the mean defined on the TRAINING set
            XtrainReadyForTraining(:,:,i) = XtrainZeroMean;
            XtestReadyForTest(:,:,i) = XtestZeroMean;
        end
        

     case 4
        NumOfSimulations = 70;
        FractionOfDataForTraining = 0.7;
        h1 = 10;
        valuesOfk = [1 2 3 4 5]; 
        NumOfEpoch_PCAsae = [];
        VE_LinearityThreshold = 99;
        
        load('...\Datasets\COIL20.mat')
        for i = 1:NumOfSimulations
            TrainIdx = randperm(size(X,1),round(size(X,1)*FractionOfDataForTraining)); 
            TestIdx = 1:size(X,1);
            TestIdx(TrainIdx) = [];
            Xtrain = X(TrainIdx,:);
            Xtest = X(TestIdx,:);
            v_trainMean = mean(Xtrain); % Stores the mean of the process
            XtrainZeroMean = Xtrain - v_trainMean;
            XtestZeroMean = Xtest - v_trainMean; % Removes the mean defined on the TRAINING set
            XtrainReadyForTraining(:,:,i) = XtrainZeroMean;
            XtestReadyForTest(:,:,i) = XtestZeroMean;
        end
        
     case 5
        NumOfSimulations = 70;
        FractionOfDataForTraining = 0.7;
        h1 = 10;
        valuesOfk = [1 2 3 4 5]; 
        NumOfEpoch_PCAsae = [];
        VE_LinearityThreshold = 99;
        
        load('...\Datasets\MNIST.mat')
        for i = 1:NumOfSimulations
            TrainIdx = randperm(size(X,1),round(size(X,1)*FractionOfDataForTraining)); 
            TestIdx = 1:size(X,1);
            TestIdx(TrainIdx) = [];
            Xtrain = X(TrainIdx,:);
            Xtest = X(TestIdx,:);
            v_trainMean = mean(Xtrain); % Stores the mean of the process
            XtrainZeroMean = Xtrain - v_trainMean;
            XtestZeroMean = Xtest - v_trainMean; % Removes the mean defined on the TRAINING set
            XtrainReadyForTraining(:,:,i) = XtrainZeroMean;
            XtestReadyForTest(:,:,i) = XtestZeroMean;
        end
end

%==========================================================================




%% TRAINING phase:=========================================================
for i = 1:NumOfSimulations
    
    actual_XtrainReadyForTraining = XtrainReadyForTraining(:,:,i);
    actual_XtestReadyForTest = XtestReadyForTest(:,:,i);
    
    [k_linTrain, Ptrain_lin, Ttrain_lin, ~, v_compTime_PCA] = define_k_lin(actual_XtrainReadyForTraining, VE_LinearityThreshold, 'PCA'); % (a) Ptrain_lin
    v_k_linTrain(i,1) = k_linTrain;
    M_compTime_PCA{i,:} = v_compTime_PCA;

    for k = valuesOfk
        % (1) PCA (Models to define in training: (a) Ptrain):
        Ptrain = pca_nipals(actual_XtrainReadyForTraining,k); % PCA wants null mean.
        
        % (2) PCAsae (Models to define in training: (a) Ptrain_lin (b) SAEtrain (c) BetaTrain_PCAsae):
        v_NumOfNeuronsPerEncoder_PCAsae = [h1+13 k]; % = [k] equivalence condition
        
        tic;
        v_NumOfNeuronsPerHiddenDecoder_PCAsae = flip(v_NumOfNeuronsPerEncoder_PCAsae(1,1:length(v_NumOfNeuronsPerEncoder_PCAsae)-1));
        SAEtrain = se(Ttrain_lin, [v_NumOfNeuronsPerEncoder_PCAsae  v_NumOfNeuronsPerHiddenDecoder_PCAsae]); % (b.1) SAEtrain (pre-trained)
        OutputLayer = feedforwardnet([]);
        if size(v_NumOfNeuronsPerEncoder_PCAsae,2) > 1
            OutputLayer = configure(OutputLayer,SAEtrain(Ttrain_lin'));
        else
            OutputLayer = configure(OutputLayer,encode(SAEtrain,Ttrain_lin'));
        end
        SAEtrain = stack(SAEtrain,OutputLayer); % Add output layer before fine-tuning
        if ~isempty(NumOfEpoch_PCAsae)
           SAEtrain.trainParam.epochs = NumOfEpoch_PCAsae; 
        end
        SAEtrain = train(SAEtrain, Ttrain_lin', Ttrain_lin'); % (b.2) SAEtrain (fine-tuned)
        Ttrain_lin_hat = SAEtrain(Ttrain_lin')';
        BetaTrain_PCAsae = pinv(Ttrain_lin_hat)*actual_XtrainReadyForTraining; % (c) BetaTrain_PCAsae
        M_compTime_PCAsae(i,find(valuesOfk==k)) = toc + M_compTime_PCA{i,:}(1,k_linTrain);
        
        % (3) RLC (Models to define in training: (a) Ptrain (b) RLCtrain (c) BetaTrain_RLC):
        v_HiddenLayers_RLC = [h1]; % = [2*k] equivalence condition
        
        tic;
        % (a) Ptrain (currently taken from PCA above)
        [RLCtrain, Ttrain_discarded_hat] = rlc_pca(Ttrain_lin, k_linTrain, k, v_HiddenLayers_RLC); % (b) RLCtrain
        BetaTrain_RLC = pinv([Ttrain_lin(:,1:k)  Ttrain_discarded_hat])*actual_XtrainReadyForTraining; % (c) BetaTrain_RLC
        M_compTime_RLC(i,find(valuesOfk==k)) = toc + M_compTime_PCA{i,:}(1,k_linTrain);
        
%==========================================================================
        
        
        
        
        
%% TEST phase:=============================================================
        % (1) PCA:
        Ttest = actual_XtestReadyForTest*Ptrain;
        actual_XtestReadyForTest_hat = Ttest*Ptrain';
        VE_Test = VarExp(actual_XtestReadyForTest, actual_XtestReadyForTest_hat);
        M_VEtest_PCA(i,find(valuesOfk==k)) = VE_Test;
        
        % (2) PCAsae:
        Ttest_lin = actual_XtestReadyForTest*Ptrain_lin; % Uses (a)
        Ttest_lin_hat = SAEtrain(Ttest_lin')'; % Uses (b)
        actual_XtestReadyForTest_hat = Ttest_lin_hat*BetaTrain_PCAsae; % Uses (c)
        VE_Test = VarExp(actual_XtestReadyForTest, actual_XtestReadyForTest_hat);
        M_VEtest_PCAsae(i,find(valuesOfk==k)) = VE_Test;
        
        % (3) RLC:
        Ttest = actual_XtestReadyForTest*Ptrain; % Uses (a)
        Ttest_discarded_hat = RLCtrain(Ttest')'; % Uses (b)
        actual_XtestReadyForTest_hat = [Ttest  Ttest_discarded_hat]*BetaTrain_RLC; % Uses (c)
        VE_Test = VarExp(actual_XtestReadyForTest, actual_XtestReadyForTest_hat);
        M_VEtest_RLC(i,find(valuesOfk==k)) = VE_Test;
%==========================================================================
        
        
        
        
        %% Just to check the code, on training set:
        % (1) PCA:
        Ttrain = actual_XtrainReadyForTraining*Ptrain;
        actual_XtrainReadyForTraining_hat = Ttrain*Ptrain';
        VE_train = VarExp(actual_XtrainReadyForTraining, actual_XtrainReadyForTraining_hat);
        M_VEtrain_PCA(i,find(valuesOfk==k)) = VE_train;
        
        % (2) PCAsae:
        Ttrain_lin = actual_XtrainReadyForTraining*Ptrain_lin; % Uses (a)
        Ttrain_lin_hat = SAEtrain(Ttrain_lin')'; % Uses (b)
        actual_XtrainReadyForTraining_hat = Ttrain_lin_hat*BetaTrain_PCAsae; % Uses (c)
        VE_train = VarExp(actual_XtrainReadyForTraining, actual_XtrainReadyForTraining_hat);
        M_VEtrain_PCAsae(i,find(valuesOfk==k)) = VE_train;
        
        % (3) RLC:
        Ttrain = actual_XtrainReadyForTraining*Ptrain; % Uses (a)
        Ttrain_discarded_hat = RLCtrain(Ttrain')'; % Uses (b)
        actual_XtrainReadyForTraining_hat = [Ttrain  Ttrain_discarded_hat]*BetaTrain_RLC; % Uses (c)
        VE_train = VarExp(actual_XtrainReadyForTraining, actual_XtrainReadyForTraining_hat);
        M_VEtrain_RLC(i,find(valuesOfk==k)) = VE_train;
    end
end
  

%% Plots:
% VE:
errorbar(valuesOfk, mean(M_VEtest_PCA),std(M_VEtest_PCA)./2, std(M_VEtest_PCA)./2,'r-o','LineWidth',4,'MarkerSize',10,'CapSize',15)
hold on;
errorbar(valuesOfk, mean(M_VEtest_PCAsae), std(M_VEtest_PCAsae)./2, std(M_VEtest_PCAsae)./2,'b-o','LineWidth',4,'MarkerSize',10,'CapSize',15)
hold on;
errorbar(valuesOfk, mean(M_VEtest_RLC), std(M_VEtest_RLC)./2, std(M_VEtest_RLC)./2,'k-.o','LineWidth',4,'MarkerSize',10,'CapSize',15)
hold on;
errorbar(valuesOfk, mean(M_VEtrain_PCA), std(M_VEtrain_PCA)./2, std(M_VEtrain_PCA)./2,'g--o','LineWidth',4,'MarkerSize',10,'CapSize',15)
hold on;
errorbar(valuesOfk, mean(M_VEtrain_PCAsae), std(M_VEtrain_PCAsae)./2, std(M_VEtrain_PCAsae)./2,'c--o','LineWidth',4,'MarkerSize',10,'CapSize',15)
hold on;
errorbar(valuesOfk, mean(M_VEtrain_RLC), std(M_VEtrain_RLC)./2, std(M_VEtrain_RLC)./2, 'm-.o','LineWidth',4,'MarkerSize',10,'CapSize',15)
hold on;
legend('PCA(OnTest)','PCAsae(OnTest)','RLC(OnTest)','PCA(OnTraining)','PCAsae(OnTraining)','RLC(OnTraining)');
ylabel('VE (%)','FontSize',30)
xlabel('Values of k','FontSize',30)
set(gca,'FontSize',30) 