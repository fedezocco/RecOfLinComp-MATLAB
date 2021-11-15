%% Implemented by Federico Zocco, last update: 15/11/2021

%% Script to generate the simulated dataset used in [1].

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

m = 500;
v = 50;
g_mean = 0;
g_std = 1;
noise_std = 0.1;
bias = 70;
NumOfRealizations = 100;
DoYouWantClasses = 0; % if = 1 (i.e. 'Yes'), the m samples are split in classes.


Psi = normrnd(0,1,10,v - 10); % To define the linear map between the 10 x_i variables and the remaining v - 10. 
for i = 1:NumOfRealizations
    g1 = normrnd(g_mean,g_std,[m,1]);
    g2 = normrnd(g_mean,g_std,[m,1]);
    g3 = normrnd(g_mean,g_std,[m,1]);
    
    x1 = sin(g1)   + normrnd(0,noise_std,[m,1]);
    x2 = cos(g1)   + normrnd(0,noise_std,[m,1]);
    x3 = x1.^7     + normrnd(0,noise_std,[m,1]);
    x4 = sign(g1).*tansig(g1)   + normrnd(0,noise_std,[m,1]);
    x5 = sqrt(abs(x3)+abs(x1))/2   + normrnd(0,noise_std,[m,1]);
    x6 = g1.*x1.^2.*x2.^3   + normrnd(0,noise_std,[m,1]);
    x7 = cos(10*g2).^3   + normrnd(0,noise_std,[m,1]);
    x8 = (abs(g2/5.^5)-exp(g2))/70   + normrnd(0,noise_std,[m,1]);
    x9 = 1./(2+abs(g1)+exp(g2))   + normrnd(0,noise_std,[m,1]);
    x10 = (g1.^3).*(g2/2.^3).*(g3/2.^3).*exp(sign(g1).*g3).*(cos(g3).^2).*sin(g3)/44   + normrnd(0,noise_std,[m,1]);
    
    XnonLin = [x1 x2 x3 x4 x5 x6 x7 x8 x9 x10];
    Xlin = XnonLin*Psi + normrnd(0,noise_std,[m,v-size(XnonLin,2)]);
    Xstore(:,:,i) = [XnonLin  Xlin] + bias*ones(size([XnonLin  Xlin]));
end
Xtrain = Xstore(:,:,1);
Xtest = Xstore(:,:,2);




%% Generator of classes (to simulate a classification scenario):
if DoYouWantClasses == 1
   classesThreshold = 0.5;
   discriminatorSample = Xtrain(1,:);
   Xtrain(1,:) = [];
   for i = 1:size(Xtrain,1)
       C = abs(corrcoef(discriminatorSample,Xtrain(i,:)));
       actualCoeff = C(1,2);
       if actualCoeff <= classesThreshold
           v_labelsTrain(i,1) = 1;
       else
           v_labelsTrain(i,1) = 2;
       end
   end
   Xtest(1,:) = [];
   for i = 1:size(Xtest,1)
       C = abs(corrcoef(discriminatorSample,Xtest(i,:)));
       actualCoeff = C(1,2);
       if actualCoeff <= classesThreshold
           v_labelsTest(i,1) = 1;
       else
           v_labelsTest(i,1) = 2;
       end
   end
end