
function [V]= VarExp(X,Xs,VT)

% (c) Seán McLoone, May 2016

if nargin <3
    VT=X(:)'*X(:);
end

B=pinv(Xs)*X;
Xhat=Xs*B;
E=Xhat-X;

V= (1- E(:)'*E(:)/VT)*100;

return
