function [Y] = lle(X,K,d)
% Locally Linear Embedding �ֲ�����Ƕ�� (ʹ��K����)
% 
% ����
%   X - ���ݾ��� D x N ���� (D - ά��, N - ������ĸ���)
%   K - ������
%   d - ���Ƕ��ά��
% ���
%   Y - Ƕ��Ϊ d x N �ľ���

[D,N] = size(X);
fprintf(1,'LLE running on %d points in %d dimensions\n',N,D);


% STEP1: �����Ӧ�����Ѱ���ھ�
fprintf(1,'-->Finding %d nearest neighbours.\n',K);

X2 = sum(X.^2,1);
distance = repmat(X2,N,1)+repmat(X2',1,N)-2*X'*X;

[sorted,index] = sort(distance);
neighborhood = index(2:(1+K),:);



% STEP2: ����ؽ�Ȩ������
fprintf(1,'-->Solving for reconstruction weights.\n');

if(K>D) 
 fprintf(1,' [note: K>D; regularization will be used]\n'); 
 tol=1e-3; % ����Լ��������£��������������ܵ�����
else
 tol=0;
end

W = zeros(K,N);
for ii=1:N
 z = X(:,neighborhood(:,ii))-repmat(X(:,ii),1,K); % �����Ƶ�ԭ��
 C = z'*z; % ���ص�Э����
 C = C + eye(K,K)*tol*trace(C); % ���� (K>D)
 W(:,ii) = C\ones(K,1); %  Cw=1
 W(:,ii) = W(:,ii)/sum(W(:,ii)); %  sum(w)=1
end;


% STEP 3: �ӳɱ����������Ƕ����� M=(I-W)'(I-W)
fprintf(1,'-->Computing embedding.\n');

% M=eye(N,N); % ʹ�� 4KN ������Ԫ�ص�ϡ�����洢
M = sparse(1:N,1:N,ones(1,N),N,N,4*K*N); 
for ii=1:N
 w = W(:,ii);
 jj = neighborhood(:,ii);
 M(ii,jj) = M(ii,jj) - w';
 M(jj,ii) = M(jj,ii) - w;
 M(jj,jj) = M(jj,jj) + w*w';
end;

% Ƕ�����
options.disp = 0; options.isreal = 1; options.issym = 1; 
[Y,eigenvals] = eigs(M,d+1,0,options);
Y = Y(:,2:d+1)'*sqrt(N); % bottom evect is [1,1,1,1...] with eval 0


fprintf(1,'Done.\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% ���������򻯷��� K>D
% C = C + tol*diag(diag(C)); % ���滯
% C = C + eye(K,K)*tol*trace(C)*K; % ���滯
