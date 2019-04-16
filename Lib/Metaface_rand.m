%
% function [D,ALPHA]=Metaface_rand(X,p,lambda,Jstep_T,Max_iteration)
%
%  This fuction initialize the Dictionary using random projection.  
%
% output:D an dictionary we want to learn with size of m by p
% input :X denote the dataset matrix with size of m by n, every column of X denote one image
% Jstep_T:  the reduction of object value in each step
% lamba: denote a coeff of l1_ls
% the formula is that : J=
% argmin(D,ALPHA){norm(X-D*ALPHA,2)^2+lamta*norm(ALPHA,1)}

function [D,ALPHA,error]=Metaface_rand(X,p,lambda,Jstep_T,Max_iteration)

n           =   size(X,2);
Jnow        =   10;
Jpre        =   0;
error       =   [];
iteration   =   0;

%initialize D using random vector
numcomps    =   p;
P           =   randn(size(X,1),numcomps);
P           =   P./ repmat(sqrt(sum(P.*P)),[size(P,1) 1]);
D           =   P;
%initialize D end

while abs(Jnow-Jpre)>Jstep_T  && iteration < Max_iteration
    fprintf(['Metaface:steperror=' num2str(Jnow-Jpre) ';and iteration=' num2str(iteration) '.\n']);
% Fix D 
for i=1:n
%    [s,status]    =  l1_ls(D, X(:,i), lambda);
   s=LeastR(D, X(:,i), lambda);
%  s=BPDN_homotopy_function(D, X(:,i), lambda, 500);
        ALPHA(:,i)     =  s;
end
%optimation one column by one begin

% Fix ALPHA, update D. begin
for i=1:p
   ai        =    ALPHA(i,:);
   Y         =    X-D*ALPHA+D(:,i)*ai;
   di        =    Y*ai';
   di        =    di./norm(di,2);
   D(:,i)    =    di;
end
% Fix ALPHA, update D. end

Jpre          =    Jnow;
zz            =    X-D*ALPHA;zalpha=ALPHA(:);
Jnow          =    zz(:)'*zz(:)+lambda*sum(abs(zalpha(:)))
iteration     =    iteration+1;
error         =    [error Jnow]

plot(error,'rs');

end