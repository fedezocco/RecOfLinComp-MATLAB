%% Implemented by Federico Zocco, last update: 15/11/2021

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


function SE = se(X, v_NumOfNeuronsPerEncoder) 


NumOfEncoders = size(v_NumOfNeuronsPerEncoder,2);
AE_1 = trainAutoencoder(X', v_NumOfNeuronsPerEncoder(1,1)); 
Z_tilde = encode(AE_1, X')';


SE_tilde = AE_1;
if NumOfEncoders > 1
    for i = 2:NumOfEncoders
        AE_i = trainAutoencoder(Z_tilde', v_NumOfNeuronsPerEncoder(1,i)); 
        SE_tilde = stack(SE_tilde, AE_i);
        %view(SE_tilde);
        Z_tilde = encode(AE_i, Z_tilde')';
    end
end
SE = SE_tilde;

