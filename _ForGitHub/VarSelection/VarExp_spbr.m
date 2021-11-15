
function [V]= VarExp(X,Xs,VT)
%
% As defined in FSCA paper -- for true variance columns of X need to have
% zero mean
%
if nargin <3
    VT=X(:)'*X(:);
end

B=pinv(Xs)*X;
Xhat=Xs*B;
E=Xhat-X;

V= (1- E(:)'*E(:)/VT)*100;

return
