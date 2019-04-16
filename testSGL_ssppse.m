clear all
clc


%%  Classifier NN, LRC and SRC    %%


pca=[0.001];

for t=1:1
    

%load AR_Expression_SSPPb50.mat   % Gallery sample: 1;  Testing sample: 3
%load AR_Illumination_SSPPb50.mat  % Gallery sample: 1; Testing sample: 3
load AR_Disguise_SSPPb50.mat        % Gallery sample: 1; Testing sample: 2

%load AR_Variance_Generic50.mat;    % Variation dictionary learned by ESRC, for comparison 
load AR_VariationDictionary_SGL.mat    % Variation dictioanry learned by SGL


dim=size(Iv,1);
class=size(Iv,3);
elltrain=1; % training sample;
elltest=2; % test sampe;

TotalTrain = class * elltrain;
TotalTest = class * elltest;

% fea: Rows of vectors of data points. Each row is x_i
Itrain1 = zeros(dim,TotalTrain);
 for i=1:class
    for j=1:elltrain
         Itrain1(:,j+(i-1)*elltrain) = Iv(:,j,i);
    end
 end
 trainlabels = constructlabel(class, elltrain);


Itrain=Itrain1;


Itest = zeros(dim,TotalTest);
for i=1:class
    for j=1:elltest
        Itest(:,j+(i-1)*elltest) = Iv(:,j+elltrain,i);
    end
end
testlabels = constructlabel(class,elltest);



par.dim = dim;
par.tr_num = TotalTrain;  % The number of training samples
par.tt_num = TotalTest;   % The number of testing samples




Igeneric=[];
Igeneric=Itrain_Variance;


% load Peal_Train_Recovery_90percent_SSPP50_4.mat
% Itrain(:,indC)=MIP;


  Itrain = Itrain ./ repmat(sqrt(sum(Itrain .* Itrain )),[size(Itrain ,1) 1]); % unit norm 2
%  Itest = Itest ./ repmat(sqrt(sum(Itest .* Itest)),[size(Itest ,1) 1]); % unit norm 2
  Igeneric = Igeneric ./ repmat(sqrt(sum(Igeneric .* Igeneric)),[size(Igeneric ,1) 1]); % unit norm 2
 
% Select Itrain' instead of Itrain, and Itest' instead of Itest 
Itrain = Itrain';
Itest = Itest';
Igeneric=Igeneric';

 %% Dimension Reduction
  %
    options = [];
    I=[Itrain;Igeneric];
    %options.ReducedDim=rank(I);
    options.ReducedDim = 300;
  [WProj, eigvalue] = PCA(I, options);
  %}     
  

%% The projected feature in subspace
FeaTrain = Itrain * WProj;
FeaTrain = FeaTrain';
FeaTest = Itest * WProj;
FeaTest = FeaTest';

 FeaGeneric=Igeneric*WProj;
 FeaGeneric=FeaGeneric';
 
 
%% Conducting classification

% %% Nearest Neighbor Classifier
% %%%----------Nearest Neighbor Classifier------------ %%%%%%
% %%Distance_mark='L2';
% %%Miss_NUM = Classifier_NN_f(FeaTrain,FeaTest,Distance_mark)  
% Miss_NUM=0;
%     Dist_Metric = 'L2';
% 
%     for i=1:par.tt_num
%         Test = FeaTest(:,i);
%         min_dist = 1e30;
%         for j=1:par.tr_num
%             Train = FeaTrain(:,j);
%             Diff = Test-Train;
%             switch Dist_Metric
%              case {'Euclidean', 'L2'}
%                   dist=norm(Diff,2); % Euclead (L2) distance
%              case 'L1'
%                   dist=norm(Diff,1); % L1 distance
%              case 'Cos'
%                   dist=acos(Test'*Train/(norm(Test,2)*norm(Train,2)));     % cos distance
%              otherwise
%                   dist=norm(Diff,2); % Default distance
%             end
%             if min_dist>dist
%                min_dist=dist;
%                Class_No=trainlabels(j);
%             end
%         end
%         if Class_No~=testlabels(i) % strncmp is to compare the first n characters of two strings
%             Miss_NUM=Miss_NUM+1;
%         end
%     end
% Miss_NUM
% Recognition_rateNN = (par.tt_num-Miss_NUM)/par.tt_num; 


%% face Identification

Distance_mark='L2';
alpha=0.01;
% lambda=pca(t);  
lambda=0.001;  % Regularization parameter


[Miss_NUMESRC1, Miss_NUMESRC2,Time] =  ExtendedSRC(FeaTrain,FeaGeneric, trainlabels,FeaTest,testlabels,lambda,Distance_mark, WProj);
Recognition_rateSGL_top1(t)=(par.tt_num-Miss_NUMESRC1)/par.tt_num;
Recognition_rateSGL_top5(t)=(par.tt_num-Miss_NUMESRC2)/par.tt_num;
end

%% Recognition Rate (NN, P+V based SRC)

Recognition_rateSGL_top1_avg=sum(Recognition_rateSGL_top1)/t
Recognition_rateSGL_top5_avg=sum(Recognition_rateSGL_top5)/t