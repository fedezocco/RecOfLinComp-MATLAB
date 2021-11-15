%% Implemented by Federico Zocco, last update: 15/11/2021

%% The script performs Monte Carlo simulations on the dataset selected by the 'switch'.
% Five methods are compared: (1) FSCA, (2) FSCA-SDE, (3) FSCA-RLC, (4) SPBR, (5) MPBR. 
% As in the name 'Simulation_VarSel_RandomSets', this
% script considers variable selection and random sets, i.e. the
% training and test sets are randomly defined at the beginning of each simulation.

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


%===================== PARAMETERS TO SET: ======================
clear all;
datasetSelector = 1;

switch datasetSelector    
    case 1
        NumOfSimulations = 100;
        FractionOfDataForTraining = 0.7;
        h1 = 6;
        valuesOfk = [1 2 3 4 5 6 7 8];
        v_NumOfNeuronsPerHiddenDecoder_FSCAsde = [h1+5 h1+15];
        v_HiddenLayers_RLC = [h1];
        setting = 'Low'; %'Low' or 'High' (Setting for datasets having either high or low value of k_lin)
        
        switch setting
            case 'Low'
                VE_LinearityThreshold = 99;
                
            case 'High' % Whenever k_lin is high: compute k_linTrain using fsca.m before running the simulation and use it here as an input
                k_linTrain = [];
        end
        
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
        NumOfSimulations = 100;
        FractionOfDataForTraining = 0.7;
        h1 = 1;
        valuesOfk = [1 2 3 4 5 6 7];
        v_NumOfNeuronsPerHiddenDecoder_FSCAsde = [h1+10 h1+20];
        v_HiddenLayers_RLC = [h1];
        setting = 'Low'; %'Low' or 'High' (Setting for datasets having either high or low value of k_lin)
        
        switch setting
            case 'Low'
                VE_LinearityThreshold = 97;
                
            case 'High' % Whenever k_lin is high: compute k_linTrain using fsca.m before running the simulation and use it here as an input
                k_linTrain = [];
        end
        
        load('...\Datasets\Xbusiness.mat')
        for i = 1:NumOfSimulations
            TrainIdx = randperm(size(Xbusiness,1),round(size(Xbusiness,1)*FractionOfDataForTraining));
            TestIdx = 1:size(Xbusiness,1);
            TestIdx(TrainIdx) = [];
            Xtrain = Xbusiness(TrainIdx,:);
            Xtest = Xbusiness(TestIdx,:);
            v_trainMean = mean(Xtrain); % Stores the mean of the process
            XtrainZeroMean = Xtrain - v_trainMean;
            XtestZeroMean = Xtest - v_trainMean; % Removes the mean defined on the TRAINING set
            XtrainReadyForTraining(:,:,i) = XtrainZeroMean;
            XtestReadyForTest(:,:,i) = XtestZeroMean;
        end
      

     case 3 % HERE DOES NOT SIMULATE (1) FSCAsde
        NumOfSimulations = 70; 
        FractionOfDataForTraining = 0.15;
        h1 = 5;
        valuesOfk = [1 2 3 4 5 6 20 30 40 50];
        v_NumOfNeuronsPerHiddenDecoder_FSCAsde = [h1+10 h1+15];
        v_HiddenLayers_RLC = [h1];
        setting = 'High'; %'Low' or 'High' (Setting for datasets having either high or low value of k_lin)
        
        switch setting
            case 'Low'
                VE_LinearityThreshold = [];
                
            case 'High' % Whenever k_lin is high: compute k_linTrain using fsca.m before running the simulation and use it here as an input
                k_linTrain = 54; % 97% of VE
                NumOfEpoch_RLC = 20;
        end
        
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
     

     case 4 % HERE DOES NOT SIMULATE (1) FSCAsde, (2) MPBR
        NumOfSimulations = 60; 
        FractionOfDataForTraining = 0.45;
        h1 = 5;
        valuesOfk = [1 2 3 4 5 6 20 40 60 80];
        v_NumOfNeuronsPerHiddenDecoder_FSCAsde = [h1+10 h1+15];
        v_HiddenLayers_RLC = [h1];
        setting = 'High'; %'Low' or 'High' (Setting for datasets having either high or low value of k_lin)
        
        switch setting
            case 'Low'
                VE_LinearityThreshold = [];
                
            case 'High' % Whenever k_lin is high: compute k_linTrain using fsca.m before running the simulation and use it here as an input
                k_linTrain = 97; % 97% of VE
                NumOfEpoch_RLC = 8;
        end
        
        load('...\Datasets\YaleB.mat')
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
        NumOfSimulations = 30; 
        FractionOfDataForTraining = 0.4;
        h1 = 3;
        valuesOfk = [1 2 3 4 5 6 8 10];
        v_NumOfNeuronsPerHiddenDecoder_FSCAsde = [h1+10 h1+15];
        v_HiddenLayers_RLC = [h1];
        setting = 'High'; %'Low' or 'High' (Setting for datasets having either high or low value of k_lin)
        
        switch setting
            case 'Low'
                VE_LinearityThreshold = [];
                
            case 'High' % Whenever k_lin is high: compute k_linTrain using fsca.m before running the simulation and use it here as an input
                k_linTrain = 20; % 99% of VE
                NumOfEpoch_RLC = 1000;
                NumOfEpoch_FSCAsde = 20;
        end
        
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
      

     case 6
        NumOfSimulations = 70;
        FractionOfDataForTraining = 0.7;
        h1 = 6;
        valuesOfk = [1 2 3 4 5 6];
        v_NumOfNeuronsPerHiddenDecoder_FSCAsde = [h1 h1+10];
        v_HiddenLayers_RLC = [h1];
        setting = 'Low'; %'Low' or 'High' (Setting for datasets having either high or low value of k_lin)
        
        switch setting
            case 'Low'
                VE_LinearityThreshold = 99.9;
                
            case 'High' % Whenever k_lin is high: compute k_linTrain using fsca.m before running the simulation and use it here as an input
                k_linTrain = [];
        end
        
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
        
        
     case 7
        NumOfSimulations = 70;
        FractionOfDataForTraining = 0.7;
        h1 = 9;
        valuesOfk = [1 2 3 4 5 6];
        v_NumOfNeuronsPerHiddenDecoder_FSCAsde = [h1 h1+10];
        v_HiddenLayers_RLC = [h1];
        setting = 'Low'; %'Low' or 'High' (Setting for datasets having either high or low value of k_lin)
        
        switch setting
            case 'Low'
                VE_LinearityThreshold = 99.9;
                
            case 'High' % Whenever k_lin is high: compute k_linTrain using fsca.m before running the simulation and use it here as an input
                k_linTrain = [];
        end
        
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

%================================================================





%% TRAINING phase:=========================================================
for i = 1:NumOfSimulations
    
    actual_XtrainReadyForTraining = XtrainReadyForTraining(:,:,i);
    actual_XtestReadyForTest = XtestReadyForTest(:,:,i);
    
    switch setting
        case 'Low'
            [k_linTrain, ~, ~, v_selectedLinTrain_FSCA, v_compTime_FSCA] = define_k_lin(actual_XtrainReadyForTraining, VE_LinearityThreshold, 'FSCA'); % (a) Linear selection
            M_compTime_FSCA{i,:} = v_compTime_FSCA;
        case 'High'
            tic;
            [~, ~, ~, v_selectedLinTrain_FSCA] = fsca(actual_XtrainReadyForTraining, k_linTrain); % (a) Linear selection
            compTime_FSCA_with_k_linTrain = toc;
    end
    
    for k = valuesOfk
            % (1) FSCA (To define in training: (a) v_selectedTrain_FSCA, (b) BetaTrain_FSCA):
            if  strcmp(setting,'High')
                tic;
                fsca(actual_XtrainReadyForTraining, k);
                M_compTime_FSCA(i,find(valuesOfk==k)) = toc;
            end
            v_selectedTrain_FSCA = v_selectedLinTrain_FSCA(1,1:k); % (a) v_selectedTrain_FSCA
            if k == valuesOfk(1,length(valuesOfk))
               M_selectedTrain_FSCA(i,:) = v_selectedTrain_FSCA;
            end
            BetaTrain_FSCA = pinv(actual_XtrainReadyForTraining(:,v_selectedTrain_FSCA))*actual_XtrainReadyForTraining; % (b) BetaTrain_FSCA
            
            % (2) SPBR (To define in training: (a) v_selectedTrain_SPBR (c) BetaTrain_SPBR):
            tic;
            v_selectedTrain_SPBR = spbr(actual_XtrainReadyForTraining, v_selectedTrain_FSCA); % (a) v_selectedTrain_SPBR
            if k == valuesOfk(1,length(valuesOfk))
               M_selectedTrain_SPBR(i,:) = v_selectedTrain_SPBR;
            end
            BetaTrain_SPBR = pinv(actual_XtrainReadyForTraining(:,v_selectedTrain_SPBR))*actual_XtrainReadyForTraining; % (b) BetaTrain_SPBR
            if strcmp(setting,'Low')
                M_compTime_SPBR(i,find(valuesOfk==k)) = toc + M_compTime_FSCA{i,:}(1,k);
            else
                M_compTime_SPBR(i,find(valuesOfk==k)) = toc + M_compTime_FSCA(i,find(valuesOfk==k));
            end
               
            % (3) MPBR (To define in training: (a) v_selectedTrain_MPBR (c) BetaTrain_MPBR):
            tic;
            v_selectedTrain_MPBR = mpbr(actual_XtrainReadyForTraining, v_selectedTrain_FSCA); % (a) v_selectedTrain_MPBR
            if k == valuesOfk(1,length(valuesOfk))
               M_selectedTrain_MPBR(i,:) = v_selectedTrain_MPBR;
            end
            BetaTrain_MPBR = pinv(actual_XtrainReadyForTraining(:,v_selectedTrain_MPBR))*actual_XtrainReadyForTraining; % (b) BetaTrain_MPBR
            if strcmp(setting,'Low')
                M_compTime_MPBR(i,find(valuesOfk==k)) = toc + M_compTime_FSCA{i,:}(1,k);
            else
                M_compTime_MPBR(i,find(valuesOfk==k)) = toc + M_compTime_FSCA(i,find(valuesOfk==k));
            end
            
            % (4) FSCAsde (To define in training: (a) v_selectedTrain_FSCA (b) SDEtrain):
            tic;
            SDEtrain = se(actual_XtrainReadyForTraining(:,v_selectedTrain_FSCA), v_NumOfNeuronsPerHiddenDecoder_FSCAsde); % (b) Pre-training of SDEtrain ((a) taken from above)
            OutputLayer = feedforwardnet([]); 
            OutputLayer = configure(OutputLayer, SDEtrain(actual_XtrainReadyForTraining(:,v_selectedTrain_FSCA)')); 
            SDEtrain = stack(SDEtrain, OutputLayer); % (b) Add the output layer of SDEtrain
            if strcmp(setting,'High')
                SDEtrain.trainParam.epochs = NumOfEpoch_FSCAsde;
            end
            SDEtrain = train(SDEtrain, actual_XtrainReadyForTraining(:,v_selectedTrain_FSCA)', actual_XtrainReadyForTraining'); % (b) Fine-tuning of SDEtrain
            if strcmp(setting,'Low')
                M_compTime_FSCAsde(i,find(valuesOfk==k)) = toc + M_compTime_FSCA{i,:}(1,k);
            else
                M_compTime_FSCAsde(i,find(valuesOfk==k)) = toc + M_compTime_FSCA(i,find(valuesOfk==k));
            end
                  
            % (5) RLC (To define in training: (a) v_selectedTrain_FSCA (b) RLCtrain (c) BetaTrain_RLC):
            tic;
            v_discardedTrain = v_selectedLinTrain_FSCA(1,k+1:k_linTrain);
            RLCtrain = feedforwardnet(v_HiddenLayers_RLC); 
            if strcmp(setting,'High')
                RLCtrain.trainParam.epochs = NumOfEpoch_RLC;
            end
            RLCtrain = train(RLCtrain, actual_XtrainReadyForTraining(:,v_selectedTrain_FSCA)', actual_XtrainReadyForTraining(:,v_discardedTrain)');% (b) RLCtrain ((a) taken from above)
            Xtrain_discarded_hat = RLCtrain(actual_XtrainReadyForTraining(:,v_selectedTrain_FSCA)')';
            BetaTrain_RLC = pinv([actual_XtrainReadyForTraining(:,v_selectedTrain_FSCA)  Xtrain_discarded_hat])*actual_XtrainReadyForTraining; % (c) BetaTrain_RLC
            if strcmp(setting,'Low')
                M_compTime_RLC(i,find(valuesOfk==k)) = toc + M_compTime_FSCA{i,:}(1,k_linTrain);
            else
                M_compTime_RLC(i,find(valuesOfk==k)) = toc + compTime_FSCA_with_k_linTrain;
            end

%==========================================================================
            
            
            
 

%% TEST phase:=============================================================
            % (1) FSCA:
            XtestPruned = actual_XtestReadyForTest(:,v_selectedTrain_FSCA); % Uses (a)
            actual_XtestReadyForTest_hat = XtestPruned*BetaTrain_FSCA; % Uses (b)
            VE_Test = VarExp(actual_XtestReadyForTest, actual_XtestReadyForTest_hat);
            M_VEtest_FSCA(i,find(valuesOfk==k)) = VE_Test;
            
            % (2) SPBR:
            XtestPruned = actual_XtestReadyForTest(:,v_selectedTrain_SPBR); % Uses (a)
            actual_XtestReadyForTest_hat = XtestPruned*BetaTrain_SPBR; % Uses (b)
            VE_Test = VarExp(actual_XtestReadyForTest, actual_XtestReadyForTest_hat);
            M_VEtest_SPBR(i,find(valuesOfk==k)) = VE_Test;
            
            % (3) MPBR:
            XtestPruned = actual_XtestReadyForTest(:,v_selectedTrain_MPBR); % Uses (a)
            actual_XtestReadyForTest_hat = XtestPruned*BetaTrain_MPBR; % Uses (b)
            VE_Test = VarExp(actual_XtestReadyForTest, actual_XtestReadyForTest_hat);
            M_VEtest_MPBR(i,find(valuesOfk==k)) = VE_Test;
            
            % (4) FSCAsde:
            XtestPruned = actual_XtestReadyForTest(:,v_selectedTrain_FSCA); % Uses (a)
            actual_XtestReadyForTest_hat = SDEtrain(XtestPruned')'; % Uses (b)
            VE_Test = VarExp(actual_XtestReadyForTest, actual_XtestReadyForTest_hat);
            M_VEtest_FSCAsde(i,find(valuesOfk==k)) = VE_Test;
             
            % (5) RLC:
            XtestPruned = actual_XtestReadyForTest(:,v_selectedTrain_FSCA); % Uses (a)
            Xtest_discarded_hat = RLCtrain(XtestPruned')'; % Uses (b) 
            actual_XtestReadyForTest_hat = [XtestPruned  Xtest_discarded_hat]*BetaTrain_RLC; % Uses (c)
            VE_Test = VarExp(actual_XtestReadyForTest, actual_XtestReadyForTest_hat);
            M_VEtest_RLC(i,find(valuesOfk==k)) = VE_Test;
%==========================================================================
            
            
            

            %% Just to check the code, on training set:
            % (2) SPBR:
%             XtrainPruned = actual_XtrainReadyForTraining(:,v_selectedTrain_SPBR); % Uses (a)
%             actual_XtrainReadyForTraining_hat = XtrainPruned*BetaTrain_SPBR; % Uses (b)
%             VE_Train = VarExp(actual_XtrainReadyForTraining, actual_XtrainReadyForTraining_hat);
%             M_VEtrain_SPBR(i,find(valuesOfk==k)) = VE_Train; 
            
            % (3) MPBR:
%             XtrainPruned = actual_XtrainReadyForTraining(:,v_selectedTrain_MPBR); % Uses (a)
%             actual_XtrainReadyForTraining_hat = XtrainPruned*BetaTrain_MPBR; % Uses (b)
%             VE_Train = VarExp(actual_XtrainReadyForTraining, actual_XtrainReadyForTraining_hat);
%             M_VEtrain_MPBR(i,find(valuesOfk==k)) = VE_Train; 
        
    end
end


%% Plots:
% VE:
if NumOfSimulations > 1
    errorbar(valuesOfk,mean(M_VEtest_FSCA),std(M_VEtest_FSCA)./2,std(M_VEtest_FSCA)./2,'r-o','LineWidth',4,'MarkerSize',10,'CapSize',15)
    hold on;
    errorbar(valuesOfk,mean(M_VEtest_SPBR),std(M_VEtest_SPBR)./2,std(M_VEtest_SPBR)./2,'c-o','LineWidth',4,'MarkerSize',10,'CapSize',15)
    hold on;
    errorbar(valuesOfk,mean(M_VEtest_MPBR),std(M_VEtest_MPBR)./2,std(M_VEtest_MPBR)./2,'m-.o','LineWidth',4,'MarkerSize',10,'CapSize',15)
    hold on;
    errorbar(valuesOfk,mean(M_VEtest_FSCAsde),std(M_VEtest_FSCAsde)./2,std(M_VEtest_FSCAsde)./2,'g--o','LineWidth',4,'MarkerSize',10,'CapSize',15)
    hold on;
    errorbar(valuesOfk,mean(M_VEtest_RLC),std(M_VEtest_RLC)./2,std(M_VEtest_RLC)./2,'k--o','LineWidth',4,'MarkerSize',10,'CapSize',15)
    % hold on;
    % plot(valuesOfk,mean(M_VEtrain_SPBR),'b-.o','LineWidth',4,'MarkerSize',10)
    % hold on;
    % plot(valuesOfk,mean(M_VEtrain_MPBR),'b-.*','LineWidth',4,'MarkerSize',10)
    % hold on;
    % legend('FSCA(OnTest)','SPBR(OnTest)','MPBR(OnTest)','FSCAsde(OnTest)','RLC(OnTest)','SPBR(OnTraining)','MPBR(OnTraining)');
    legend('FSCA(OnTest)','SPBR(OnTest)','MPBR(OnTest)','FSCAsde(OnTest)','RLC(OnTest)');
    ylabel('VE (%)','FontSize',30)
    xlabel('Values of k','FontSize',30)
    set(gca,'FontSize',30)
else
    plot(valuesOfk,M_VEtest_FSCA,'r-o','LineWidth',4,'MarkerSize',10)
    hold on;
    plot(valuesOfk,M_VEtest_SPBR,'c-o','LineWidth',4,'MarkerSize',10)
    hold on;
    plot(valuesOfk,M_VEtest_MPBR,'m-.o','LineWidth',4,'MarkerSize',10)
    hold on;
    plot(valuesOfk,M_VEtest_FSCAsde,'g--o','LineWidth',4,'MarkerSize',10)
    hold on;
    plot(valuesOfk,M_VEtest_RLC,'k--o','LineWidth',4,'MarkerSize',10)
    hold on;
    legend('FSCA(OnTest)','SPBR(OnTest)','MPBR(OnTest)','FSCAsde(OnTest)','RLC(OnTest)');
    ylabel('VE (%)','FontSize',30)
    xlabel('Values of k','FontSize',30)
    set(gca,'FontSize',30)
end