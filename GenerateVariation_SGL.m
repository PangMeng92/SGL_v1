clear all;
clc;

 path(path, './Optimization');  
 path (path, './l1_ls_matlab');


%% AR database
load  AR_GenericSet_50.mat   % Generic set contamining 50 subjects on AR


trainlabels = Tr_Label;
NewTrain_DAT = AR_Tr;
par.nClass      =    50;    % for AR     training: 50*13
tr_dat   =   double(NewTrain_DAT(:,trainlabels<=par.nClass));
tr_lab    =   trainlabels(trainlabels<=par.nClass);
par.tr_num = size(tr_dat,2);
par.tr_each = par.tr_num / par.nClass;

par.lambda_l     =    0.001;          % parameter of l1_ls in learning  (0.001)
par.lambda_t     =    0.001;          % parameter of l1_ls in testing     (0.001)
par.dim          =    rank(tr_dat);            % the svd (pca) dimension
par.ncol_ratio   =    0.95;           % the column number of Dictionary i divided by the training sample number in class i.
par.objT         =    1e-3;           % the objective gap of metaface learning
par.nIter        =    15;             % the maximal iteration number of metaface learnng (15)


[disc_set,disc_value,Mean_Image]  =  Eigenface_f(double(tr_dat),par.dim);   % SVD is also feasible
tr_gdat        =    disc_set'* tr_dat;
tr_gdat        =    tr_gdat./ repmat(sqrt(sum(tr_gdat.*tr_gdat)),[size(tr_gdat,1) 1]); % unit norm 2


%% Metaface Dictionary Learning
for class=1:par.nClass
    tr_gdat_each   =   tr_gdat(:,(tr_lab==class));
    par.ncol   =   floor(par.ncol_ratio*size(tr_gdat_each,2));
%     [D(class).d,ALPHA(class).alpha] = Metaface(tr_gdat_each, par.ncol, par.lambda_l, par.objT, par.nIter);
	[Dict(:,(class-1)*par.ncol+1: class*par.ncol), Coeff((class-1)*par.ncol+1:class*par.ncol,:), error] = Metaface_rand(tr_gdat_each,par.ncol,par.lambda_l,par.objT,par.nIter);
    %[D(class).d,ALPHA(class).alpha]=Metaface_rand(tr_gdat_each,par.ncol,par.lambda_l,par.objT,par.nIter);
end


D=Dict;


%% Fisher information-based Feature Regrouping

par.ncol         =    size(D,2);      % the number of atoms in the Dictionary. 

%==================spare coding for each sample==========
coefficients=[];
coefficientsTest=[];

for i=1:length(tr_lab) 
    %[coeff,status] = l1_ls(D, tr_gdat(:,i), par.lambda_l); % learn the sparse coefficients for each sample
     [coeff] = BPDN_homotopy_function(D, tr_gdat(:,i), par.lambda_l, 1000);
    coefficients(:,i)=coeff;           
end

%===================feature selection======================================
feature_variance=fea_super(par.ncol,par.tr_num,coefficients,D,par.dim,par.tr_each,par.nClass);


% ==================feature regrouping============================

par.regu        = [0.5];    % Setting the threshold to separate the MDP and LDP 
Itrain_Variance=[];
MDP=[];
[a,ind]=sort(feature_variance);

coff_temp1=zeros(par.ncol,par.tr_num);
coff_temp2=zeros(par.ncol,par.tr_num);


for t=1: size(par.regu,2)
temp=round(length(ind)*par.regu(t));
coff_temp1(ind((temp+1):par.ncol),:)=coefficients(ind((temp+1):par.ncol),:);
coff_temp2(ind(1:temp),:)=coefficients(ind(1:temp),:);
tr_gdat_1=D*coff_temp1;    %MDP
tr_gdat_2=D*coff_temp2;    %LDP

% MDP and LDP
MDP_fea=disc_set*tr_gdat_1;
LDP_fea=disc_set*tr_gdat_2;

Itrain_Variance=[Itrain_Variance, LDP_fea];
MDP=[MDP, MDP_fea];
end


%% Test Original image and its MIP and LIP
i=10;
A=tr_dat(:,i);
A=mat2gray(A);
A=reshape(A,48,48);
A=A';
imshow(A);

B=MDP_fea(:,i);
B=mat2gray(B);
B=reshape(B,48,48);
B=B';
figure; imshow(B);

C=LDP_fea(:,i);
C=mat2gray(C);
C=reshape(C,48,48);
C=C';
figure; imshow(C);
