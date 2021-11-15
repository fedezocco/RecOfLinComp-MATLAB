function [S, M, VarEx, compId]=FSCA(X, Nc);

% [M, F, VarEx, CompId]=FSCA(X, Nc);
% X = n x m matrix (measurements x variables)
% NB: X is assumed to be normalised -- i.e. mean of each column is zero.
% Nc = number FSC components to compute
%
% X ~= M*F'; where M are FS components and F is the linear weightings
% VarEx gives the accumulative variance explained and CompId lists the
% coordinates of the corresponding variables.
%
% Computes FSC decompostion by selecting successive components as those
% which explain the most variance across all the data after the contributions
% of the previously selected components has been removed.
%
% Note: This is equivalent to selecting successive components as those
% which in combination with the previously selected components explain the
% most variance across all the data. (i.e the implementation of FSCA given in FSCAV)
% 
% (c) Sean McLoone, Sept 2008  - updated 2014

% if max(abs(mean(X))) > 10^-6
%     fprintf('\nWarning: Data not zero mean .. detrending\n');
%     [X, mX]= pca_normalise(X);
% end

L=size(X,2);


Y=X;
VT=var(Y(:));
compId=[];
VarEx=[];
YhatP=0;
S=[];
M=[];

if nargin <2
    Nc=1;
end
PVE=zeros(Nc,1);
for j=1:Nc
    EFS=ones(L,1)*100;
    for i=1:L;
        x=Y(:,i);
        r=Y'*x;
        EFS(i)=r'*r/(x'*x);  %Rayleigh quotient for x_i
    end
    [v Id]=max(EFS);
    x=Y(:,Id);
 
    %Deflate matrix for selected x
    th=pinv(x)*Y;
    Yhat= x*th;
    Y=Y-Yhat;
    
    S=[S x];
    M=[M  th'];
    YhatP=YhatP+Yhat; 
    compId=[compId Id];
    VarEx=[VarEx var(YhatP(:))/VT*100];

end


