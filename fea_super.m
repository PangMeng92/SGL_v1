function feature_variance=fea_super(parncol,partr_num,coefficients,D, pardim, parno_each, parnClass)

for i=1:parncol
    coff_temp=zeros(parncol,partr_num);
    coff_temp(i,:)=coefficients(i,:);
    CM=D*coff_temp;
	
    CMD=reshape(CM,[pardim,parno_each,parnClass]);
    class_mean=mean(CMD,2);  % seek for the mean belong to the cth class of a given feature, i.e., Xz,c
	overall_mean=mean(class_mean,3); % seek for the mean of the total samples of a given feature, i.e., Xz
	
	Bv=sum((class_mean-repmat(overall_mean,[1,1,parnClass])).^2,3);  % between class viarance

	Wv_each = sum((CMD-repmat(class_mean,[1,parno_each])).^2,1); % within class viarance for each class
	Wv = sum(Wv_each,3);	 % within class viarance for all classes
	Wv=Wv+1e-16;	
	
	Dv=sum(Bv)/sum(Wv);
    feature_variance(i)=Dv;  
end