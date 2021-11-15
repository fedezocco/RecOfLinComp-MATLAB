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


function [k_lin, P_lin, T_lin, v_selectedIdx, v_compTime] = define_k_lin(X, VE_threshold, selector)  % P and T in output are for the case k = k_lin (k_lin is the k that linearily satisfies the VE_threshold requested)

k_lin = 0;
VE_current = 0;

switch selector
    case 'PCA'
        while VE_current < VE_threshold
            k_lin = k_lin + 1;
            tic;
            [P_lin,T_lin,v_VE_current] = pca_nipals(X,k_lin);
            v_compTime(1,k_lin) = toc;
            VE_current = v_VE_current(k_lin);
        end
        v_selectedIdx = [];
    
    case 'FSCA'
        while VE_current < VE_threshold
            k_lin = k_lin + 1;
            tic;
            [T_lin,P_lin,v_VE_current,v_selectedIdx] = fsca(X,k_lin);
            v_compTime(1,k_lin) = toc; 
            VE_current = v_VE_current(k_lin);
        end
end
        