function [eigvector, eigvalue] = NPE(options, data)
% NPE: Neighborhood Preserving Embedding  邻域保持嵌入
%             输入:
%               data    - 数据矩阵，每个行向量是一个数据点
%               options - 以下设置:
%                      NeighborMode -  指示如何构造图形
%                           'KNN'     -  当且仅当它们属于彼此的k个邻近区域时，在两个节点之间设置边缘。默认选项。
%                           'Supervised'  -  监督，两个选项:
%                                       1. k = 0，当且仅当它们属于同一个类时，在两个节点之间放置一个边。
%                                       2. k> 0，同一类中两个节点之间的距离小于两个节点的diff。标签应提供标签信息'gnd'。每个人脸数据的标签矢量，从1，2，……，n。
%                       k           -   邻居数量。 默认 k = 5;
%                       gnd         -   NeighborMode 下的 'Supervised'所需的参数。每个数据点的标签信息的列向量。
%             输出:
%               eigvector - 每列都是嵌入函数, y = x*eigvector 将是x的嵌入结果。特征向量（每一列）
%               eigvalue  - LPP特征问题的特征值。 从最小到最大排序。

if (~exist('options','var'))
   options = [];
end

if ~isfield(options,'NeighborMode')
    options.NeighborMode = 'KNN';
end

if ~isfield(options,'k') 
    options.k = 5;
end



[nSmp,nFea] = size(data);%(行，列)

if options.k >= nSmp
    error('k is too large!');
end

if(options.k > nFea)
    tol=1e-3; % 在受约束的情况下，常规限制器会受到限制
else
    tol=1e-12;
end


% 第一步：构造近邻图。使用K-近邻法寻找与数据点欧氏距离最近的K个近邻点。
if options.k <= 0  % Always supervised!
    if ~isfield(options,'gnd')
        error('gnd should be provided!');
    end
    if length(options.gnd) ~= nSmp
        error('gnd and data mismatch!');
    end
    if ~isfield(options,'bEigs')
        options.bEigs = 0;
    end
% 第二步：确定权值。
    W = zeros(nSmp,nSmp);
    for ii=1:nSmp
        idx = find(options.gnd==options.gnd(ii));
        idx(find(idx==ii)) = [];
        z = data(idx,:)-repmat(data(ii,:),length(idx),1); % 将像素转移到原点
        C = z*z';                                   % 局部协方差
        C = C + eye(size(C))*tol*trace(C);                   % 正规化
        tW = C\ones(length(idx),1);                           % solve Cw=1
        tW = tW/sum(tW);                  % enforce sum(w)=1
        W(idx,ii) = tW;
    end
    M = (eye(size(W)) - W);
    M = M*M';
    M = max(M,M');
    M = sparse(M);
else
% 第一步：构造近邻图。使用K-近邻法寻找与数据点欧氏距离最近的K个近邻点。
    switch lower(options.NeighborMode)
        case {lower('KNN')}

            Distance = EuDist2(data,[],0); %欧氏距离
            [sorted,index] = sort(Distance,2);%排序
            neighborhood = index(:,2:(1+options.k));%找到前K个
            
        case {lower('Supervised')}
            if ~isfield(options,'gnd')
                error('gnd should be provided!');
            end
            if length(options.gnd) ~= nSmp
                error('gnd and data mismatch!');
            end
            if ~isfield(options,'bEigs')
                options.bEigs = 0;
            end
%每个人脸数据的标签矢量
            Label = unique(options.gnd);
            nLabel = length(Label);
            neighborhood = zeros(nSmp,options.k);
            for idx=1:nLabel
                classIdx = find(options.gnd==Label(idx));
                if options.k >= length(classIdx)
                    error('k is too large!');
                end
                Distance = EuDist2(data(classIdx,:),[],0); %欧氏距离
                [sorted,index] = sort(Distance,2);%排序
                neighborhood(classIdx,:) = classIdx(index(:,2:(1+options.k)));%找到前K个
            end
        otherwise
            error('NeighborMode does not exist!');
    end
% 第二步：确定权值。用近邻对各个数据点进行重构。
    W = zeros(options.k,nSmp);
    for ii=1:nSmp
        z = data(neighborhood(ii,:),:)-repmat(data(ii,:),options.k,1); % 将像素转移到原点
        C = z*z';                                        % 局部协方差
        C = C + eye(size(C))*tol*trace(C);                   % 正规化
        W(:,ii) = C\ones(options.k,1);                           % Cw=1
        W(:,ii) = W(:,ii)/sum(W(:,ii));                  % sum(w)=1
    end

    M = sparse(1:nSmp,1:nSmp,ones(1,nSmp),nSmp,nSmp,4*options.k*nSmp);%稀疏
    for ii=1:nSmp
        w = W(:,ii);
        jj = neighborhood(ii,:)';
        M(ii,jj) = M(ii,jj) - w';
        M(jj,ii) = M(jj,ii) - w;
        M(jj,jj) = M(jj,jj) + w*w';
    end
    M = max(M,M');
    M = sparse(M);
end


% 第三步：计算特征映射。
%==========================
% 如果数据量过大，注释以下代码
%==========================
if isfield(options,'keepMean') && options.keepMean
else
    if issparse(data)
        data = full(data);
    end
    sampleMean = mean(data);
    data = (data - repmat(sampleMean,nSmp,1));
end
%==========================

M = -M;
for i=1:size(M,1)
    M(i,i) = M(i,i) + 1;
end



[eigvector, eigvalue] = LGE(M, [], options, data);%调用LGE函数，计算高维数据data的线性图形嵌入


eigIdx = find(eigvalue < 1e-10);% 取特征值小于1*10^(-10)所对应的特征值
eigvalue (eigIdx) = [];% 将特征值小于1*10^(-10)的特征值清空
eigvector(:,eigIdx) = [];% 将特征值小于1*10^(-10)的特征向量去掉




