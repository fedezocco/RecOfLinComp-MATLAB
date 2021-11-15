function [IDout, VExp, MPcount]=mpbr(X, IDin)

% [IDout, rc]=mpbr(X, IDin)
%
% Implenents multi-pass backword refinement
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



compIDOrig=IDin;

if Nc==1
     IDout=IDin;
     VExp=VarExp_spbr(X,X(:,compIDOrig));
     MPcount=0;
     return
end

Rflag=1;
MPcount=0;

L=(10^-3)*eye(Nc);

while Rflag>0
    refcount=0;
    compID=compIDOrig;
    
    Z=X(:, compID);
    Cinv=inv(Z'*Z+L); 
    
    VE=0;
    for v=1:1:Nv
        q=Z'*X(:,v);
        VE=VE+q'*Cinv*q;
    end
    VEnew=VE;
    
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
        IndexSet(find(IndexSet==Newj))=compIDOrig(j);
        refcount=refcount+cnt;
    end
    
    IDout=compID;
    VExp=VarExp_spbr(X,X(:,compID));
    compIDOrig=compID;
   
    MPcount=MPcount+1;
    if refcount<1
        Rflag=0;
    end
end




