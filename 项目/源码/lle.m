function [Y] = lle(X,K,d)
% Locally Linear Embedding 局部线性嵌入 (使用K近邻)
% 
% 输入
%   X - 数据矩阵， D x N 矩阵 (D - 维数, N - 样本点的个数)
%   K - 近邻数
%   d - 最大嵌入维数
% 输出
%   Y - 嵌入为 d x N 的矩阵

[D,N] = size(X);
fprintf(1,'LLE running on %d points in %d dimensions\n',N,D);


% STEP1: 计算对应距离和寻找邻居
fprintf(1,'-->Finding %d nearest neighbours.\n',K);

X2 = sum(X.^2,1);
distance = repmat(X2,N,1)+repmat(X2',1,N)-2*X'*X;

[sorted,index] = sort(distance);
neighborhood = index(2:(1+K),:);



% STEP2: 解决重建权重问题
fprintf(1,'-->Solving for reconstruction weights.\n');

if(K>D) 
 fprintf(1,' [note: K>D; regularization will be used]\n'); 
 tol=1e-3; % 在受约束的情况下，常规限制器会受到限制
else
 tol=0;
end

W = zeros(K,N);
for ii=1:N
 z = X(:,neighborhood(:,ii))-repmat(X(:,ii),1,K); % 将点移到原点
 C = z'*z; % 当地的协方差
 C = C + eye(K,K)*tol*trace(C); % 正则化 (K>D)
 W(:,ii) = C\ones(K,1); %  Cw=1
 W(:,ii) = W(:,ii)/sum(W(:,ii)); %  sum(w)=1
end;


% STEP 3: 从成本矩阵的特征嵌入计算 M=(I-W)'(I-W)
fprintf(1,'-->Computing embedding.\n');

% M=eye(N,N); % 使用 4KN 个非零元素的稀疏矩阵存储
M = sparse(1:N,1:N,ones(1,N),N,N,4*K*N); 
for ii=1:N
 w = W(:,ii);
 jj = neighborhood(:,ii);
 M(ii,jj) = M(ii,jj) - w';
 M(jj,ii) = M(jj,ii) - w;
 M(jj,jj) = M(jj,jj) + w*w';
end;

% 嵌入计算
options.disp = 0; options.isreal = 1; options.issym = 1; 
[Y,eigenvals] = eigs(M,d+1,0,options);
Y = Y(:,2:d+1)'*sqrt(N); % bottom evect is [1,1,1,1...] with eval 0


fprintf(1,'Done.\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% 其他的正则化方法 K>D
% C = C + tol*diag(diag(C)); % 正规化
% C = C + eye(K,K)*tol*trace(C)*K; % 正规化
