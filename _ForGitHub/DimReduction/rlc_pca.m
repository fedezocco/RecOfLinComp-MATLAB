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


function [RLC, T_discarded_hat] = rlc_pca(T_lin, k_lin, k, v_HiddenLayers)

if k < k_lin
    T = T_lin(:,1:k);
    T_discarded = T_lin(:,k+1:k_lin);
    RLC = feedforwardnet(v_HiddenLayers);
    RLC = train(RLC,T',T_discarded');
    T_discarded_hat = RLC(T')';
else
    disp('No need to recover the linear components with RLC, PCA is enough')
    RLC = [];
    T_discarded_hat = [];
end