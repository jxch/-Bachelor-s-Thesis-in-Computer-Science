function [eigvector, eigvalue] = OIsoP(options, data)
% OIsoP: Orthogonal Isometric Projection 正交等距映射算法
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



INFratio = 1000;


if (~exist('options','var'))
   options = [];
end

if ~isfield(options,'NeighborMode')
    options.NeighborMode = 'KNN';
end

if ~isfield(options,'k') 
    options.k = 5;
end

nSmp = size(data,1);

if options.k >= nSmp
    error('k is too large!');
end


if options.k <= 0  % Always supervised!
    if ~isfield(options,'gnd')
        error('gnd should be provided!');
    end
    if length(options.gnd) ~= nSmp
        error('gnd and data mismatch!');
    end
    
    Label = unique(options.gnd);
    nLabel = length(Label);
% 使用K-近邻方法构造近邻图G
    G = zeros(nSmp,nSmp);
    for i=1:nLabel
        classIdx = find(options.gnd==Label(i));
        D = EuDist2(data(classIdx,:),[],1);
        G(classIdx,classIdx) = D;
    end
    maxD = max(max(G));
    INF = maxD*INFratio;  % 有效无限距离
    
    D = INF*ones(nSmp,nSmp);
    for i=1:nLabel
        classIdx = find(options.gnd==Label(i));
        D(classIdx,classIdx) = G(classIdx,classIdx);
    end
    
    clear G
else
    switch lower(options.NeighborMode)
        case {lower('KNN')}
            D = EuDist2(data);
            maxD = max(max(D));
            INF = maxD*INFratio;  % 有效无限距离
            
            [dump,iidx] = sort(D,2);
            iidx = iidx(:,(2+options.k):end);
            for i=1:nSmp
                D(i,iidx(i,:)) = 0;
            end
            D = max(D,D');
            % 使用迪杰斯特拉算法计算最短路径矩阵D
            D = sparse(D);
            D = dijkstra(D, 1:nSmp);

            D = reshape(D,nSmp*nSmp,1);
            infIdx = find(D==inf);
            if ~isempty(infIdx)
                D(infIdx) = INF;
            end
            D = reshape(D,nSmp,nSmp);

        case {lower('Supervised')}
            if ~isfield(options,'gnd')
                error('gnd should be provided!');
            end
            if length(options.gnd) ~= nSmp
                error('gnd and data mismatch!');
            end

            Label = unique(options.gnd);
            nLabel = length(Label);

% 使用K-近邻方法构造近邻图G
            G = zeros(nSmp,nSmp);
            maxD = 0;
            for idx=1:nLabel
                classIdx = find(options.gnd==Label(idx));
                nSmpClass = length(classIdx);
                D = EuDist2(data(classIdx,:),[],1);
                if maxD < max(max(D))
                    maxD = max(max(D));
                end
                if options.k >= nSmpClass
                    G(classIdx,classIdx) = D;
                else
                    [dump,iidx] = sort(D,2);
                    iidx = iidx(:,(2+options.k):end);
                    for i=1:nSmpClass
                        D(i,iidx(i,:)) = 0;
                    end
                    % 使用迪杰斯特拉算法计算最短路径矩阵D
                    D = max(D,D');
                    D = sparse(D);
                    D = dijkstra(D, 1:nSmpClass);
                    G(classIdx,classIdx) = D;
                end
            end
            
            INF = maxD*INFratio;  % 有效无限距离

            D = INF*ones(nSmp,nSmp);
            for i=1:nLabel
                classIdx = find(options.gnd==Label(i));
                D(classIdx,classIdx) = G(classIdx,classIdx);
            end
            clear G

        otherwise
            error('NeighborMode does not exist!');
    end
end


S = D.^2;
sumS = sum(S);
H = sumS'*ones(1,nSmp)/nSmp;
TauDg = -.5*(S - H - H' + sum(sumS)/(nSmp^2));

TauDg = max(TauDg,TauDg');


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

% TauDg=Schmidt_orthogonalization(TauDg);
[eigvector, eigvalue] = LGE(TauDg, [], options, data);%调用LGE函数，计算高维数据data的线性图形嵌入


eigIdx = find(eigvalue < 1e-3);% 取特征值小于1*10^(-10)所对应的特征值
eigvalue (eigIdx) = [];% 将特征值小于1*10^(-10)的特征值清空
eigvector(:,eigIdx) = [];% 将特征值小于1*10^(-10)的特征向量去掉



