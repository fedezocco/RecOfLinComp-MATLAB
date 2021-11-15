function [P, T, VE]= pca_nipals(X,Npc)

% [P, T, VE]=pca_nipals(X,Npc)
%
% Principal Component Decomposition of X, i.e.  X=T*P'
% where X is X = m measurements x v variables
% Npc is the max number of principal components to include in the
% decomposition - if not specified defaults to r =rank(X).
% T = scores matrix  (m x min(Npc,r)
% P = loading matrix (v x min(Npc,r)
% VE = Variability explained
%
% NB: This assumes X = m x v , 
% where m= no of samples and v = no of variables
%
% (c) Sean McLoone, April 2009


r=rank(X);
if nargin <2
    Npc=r;
end

if Npc > r
    Npc=r;
end

vtot= norm(X,'fro');  % total variation in the data

% select the first non-zero column of X as t0 (initial guess for t)
x = X(:,1);  cn=1;
while x'*x < 5*eps & cn < size(X,2);
    cn=cn+1;
    x=X(:,cn);
end
if x'*x < 5*eps   
    error('Data matrix has no non-zero columns')
    return
end

i=1;  %PC counter
while i<= Npc
    
    t = x;
    p = X'*t;
    p = p/norm(p,2);
    tnew = X*p;
    
    % calculate the largest component of x;
    while norm(tnew-t,2)> 1e-6
        t = tnew;
        p = X'*t;
        p = p/norm(p,2);
        tnew = X*p;
    end
    
    % save the largest component th & pht to t(scores) &
    % pt(trans(p),p:loading) 
    T(:,i) = t;
    P(:,i) = p;
    
    % calculate the residual of x when minus t*trans(p).
    
    X = X - t*p';
    
    % calculate proportion of variation in x that has been explained by ith
    % PC
    
    VE(i) = norm(X,'fro')/vtot;
    
    % preparatio for the next loop
    
    i=i+1;
    
    % select the first non-zero column of X as t0 (initial guess for t)
    x = X(:,1); cn=1;
    while x'*x < 0.1*eps & cn < size(X,2);
        cn=cn+1;
        x=X(:,cn);
    end
    if x'*x < 5*eps & i <p  
        error('Data matrix has no non-zero columns')
        return
    end   
end

VE=VE.*VE*100;
VE=100-VE;

