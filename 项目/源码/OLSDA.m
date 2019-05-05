function [eigvector, eigvalue, bSuccess] = OLSDA(gnd, options, SampleData)
% OLSDA——正交局部敏感辨别分析方法
%[eigvector, eigvalue] = OLSDA(gnd, options, data)
% SampleData - 原始数据矩阵，有限元分析的每个行向量是一个数据点。
% gnd - 每个人脸数据的标签矢量，从1，2，……，n。
% eigvector  - 映射矩阵
% eigvalue  - 映射矩阵对应的特征值
 
if (~exist('options','var'))   %若图像数据为空
   options = [];
end
 
[nSam,nCol] = size(SampleData);    %计算原始数据点矩阵的行和列数
if length(gnd) ~= nSam            %若原始数据点矩阵的行数与标签矢量不相等
    error('gnd and data mismatch!');   %则返回错误
end
 
k = 0;
if isfield(options,'k') && (options.k < nSam-1)
    %判断输入的K值是否为结构体数组options的域（成员）
    %且k的值应小于原始数据点矩阵的列数减1
    k = options.k;
end
 
beta = 0.1;    %用于优化类内图与类间图的参数，beta属于[0,1]，默认值为0.1
if isfield(options,'beta') && (options.beta > 0) && (options.beta < 1)
    %判断输入的beta值是否为结构体数组options的域（成员）
    %且beta的值应控制在0与1之间
    beta = options.beta;
end
 
Tag = unique(gnd);     %求训练样本的类别矢量
nTag = length(Tag);    %nLabel为类别矢量的长度，即类别数
 
Ww = zeros(nSam,nSam);     %类间权重矩阵
Wb = ones(nSam,nSam);      %类内权重矩阵
for idx=1:nTag     %对各个类别的数据进行分类处理
    classIdx = find(gnd==Tag(idx));
    Ww(classIdx,classIdx) = 1;
    Wb(classIdx,classIdx) = 0;
end
 
if k > 0    %当k的值大于0时
    D = EuDist2(SampleData,[],0);     % 计算欧氏距离
    [dump idx] = sort(D,2);           % 对每行进行排列
    clear D dump
    idx = idx(:,1:options.k+1);
    G = sparse(repmat([1:nSam]',[options.k+1,1]),...
    idx(:),ones(prod(size(idx)),1),nSam,nSam);    %提取矩阵中的非0元素
    G = max(G,G');
    Ww = Ww.*G;     %重构权重矩阵
    Wb = Wb.*G;
    clear G
end
 
Db = full(sum(Wb,2));    %构造非稀疏矩阵
Wb = -Wb;
for i=1:size(Wb,1)
    Wb(i,i) = Wb(i,i) + Db(i);     %两个矩阵进行相加运算
end
 
D = full(sum(Ww,2));     %构造非稀疏矩阵
 
if isfield(options,'Regu') && options.Regu     %Regu的默认值为1
    options.ReguAlpha = options.ReguAlpha*sum(D)/length(D);
end
 
W = sparse((beta/(1-beta))*Wb+Ww);
clear Wb Ww
 
% 如果输入样本数据太大，则执行以下代码
if isfield(options,'keepMean') && options.keepMean   %判断keepMean参数的值域
else
    if issparse(SampleData)              %若所输入样本数据为稀疏矩阵
        SampleData = full(SampleData);   %则将样本数据转换成非稀疏矩阵
    end
    sampleMean = mean(SampleData);               %计算样本数据的平均值
    SampleData = (SampleData - repmat(sampleMean,nSam,1));  %重构输入样本数据
end
 
if ~isfield(options,'Regu') || ~options.Regu     %判断Regu参数的值域
    DToPowerHalf = D.^.5;
    D_mhalf = DToPowerHalf.^-1;
    if nSam < 5000   %若样本数据矩阵的类书小于5000，即为小样本数据，则……
        tmpD_mhalf = repmat(D_mhalf,1,nSam);
        W = (tmpD_mhalf.*W).*tmpD_mhalf';
        clear tmpD_mhalf;
    else   %若所输入样本数据过大，则……
        [i_idx,j_idx,v_idx] = find(W);    %返回其位置
        v1_idx = zeros(size(v_idx));
        for i=1:length(v_idx)
            v1_idx(i) = v_idx(i)*D_mhalf(i_idx(i))*D_mhalf(j_idx(i));
        end
        W = sparse(i_idx,j_idx,v1_idx);     %取其非零元素
        clear i_idx j_idx v_idx v1_idx
    end
    W = max(W,W');
    SampleData = repmat(DToPowerHalf,1,nCol).*SampleData;%重构矩阵
    [eigvector, eigvalue, bSuccess] = OLGE(W, [], options, SampleData);
    %调用LGE函数，计算高维数据data的线性图形嵌入
else
    D = sparse(1:nSam,1:nSam,D,nSam,nSam); 
    [eigvector, eigvalue, bSuccess] = OLGE(W, D, options, SampleData);
    %调用LGE函数，计算高维数据data的线性图形嵌入
end
 
eigIdx = find(eigvalue < 1e-10);  % 取特征值小于1*10^(-10)所对应的特征值
eigvalue (eigIdx) = [];          % 将特征值小于1*10^(-10)的特征值清空
eigvector(:,eigIdx) = [];         % 将特征值小于1*10^(-10)的特征向量去掉


