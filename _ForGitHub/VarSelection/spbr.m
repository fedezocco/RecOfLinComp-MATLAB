function [IDout, VEnew, refcount]=spbr(X, IDin)

% [IDout, rc]=spbr(X, IDin)
%
% Implenents single-pass backword refinement
%
% X = n x m matrix (measurements x variables)
% NB: X is assumed to be normalised -- i.e. mean of each column is zero.
% IDin= index of current set of columns form X (e.g. output of FSCA analysis) 
% IDout = updated index set folliwing single pass backward refinement
% rc = refinement count (number of variables changed as a results of the
% refinement process
%
% (c) Seán McLoone, May 2016



Nv=size(X,2);
Nc= length(IDin);
refcount=0;
    
compID=IDin;
Z=X(:, compID);

L=(10^-3)*eye(Nc);
Cinv=inv(Z'*Z+L); % regularisation step to address potential singularity issues
        
VE=0;
for v=1:1:Nv
    q=Z'*X(:,v);
     VE=VE+q'*Cinv*q;
end

VEnew=VE;

if Nc==1
    IDout=IDin;
    VEnew=VarExp_spbr(X,X(:,compID));
    return
end

IndexSet=1:1:Nv;

IndexSet(compID)=[];


for j=1:1:(Nc-1)

    Newj=compID(j);
    
    cnt=0;
    for i=IndexSet
        
 
        compID(j)=i;
        Z=X(:, compID);
        Cinv=inv(Z'*Z+L);
        
        VE=0;
        for v=1:1:Nv
            q=Z'*X(:,v);
            VE=VE+q'*Cinv*q;
        end
        
        if VE > VEnew
            Newj=i;
            VEnew=VE;
            cnt=1;            
        end
    end
    compID(j)=Newj;
    IndexSet(find(IndexSet==Newj))=IDin(j);
    refcount=refcount+cnt;
end
    
   IDout=compID;
   VEnew=VarExp_spbr(X,X(:,compID));
        
   


