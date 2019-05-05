function [eigvector, eigvalue, bSuccess] = OLSDA(gnd, options, SampleData)
% OLSDA���������ֲ����б���������
%[eigvector, eigvalue] = OLSDA(gnd, options, data)
% SampleData - ԭʼ���ݾ�������Ԫ������ÿ����������һ�����ݵ㡣
% gnd - ÿ���������ݵı�ǩʸ������1��2��������n��
% eigvector  - ӳ�����
% eigvalue  - ӳ������Ӧ������ֵ
 
if (~exist('options','var'))   %��ͼ������Ϊ��
   options = [];
end
 
[nSam,nCol] = size(SampleData);    %����ԭʼ���ݵ������к�����
if length(gnd) ~= nSam            %��ԭʼ���ݵ������������ǩʸ�������
    error('gnd and data mismatch!');   %�򷵻ش���
end
 
k = 0;
if isfield(options,'k') && (options.k < nSam-1)
    %�ж������Kֵ�Ƿ�Ϊ�ṹ������options���򣨳�Ա��
    %��k��ֵӦС��ԭʼ���ݵ�����������1
    k = options.k;
end
 
beta = 0.1;    %�����Ż�����ͼ�����ͼ�Ĳ�����beta����[0,1]��Ĭ��ֵΪ0.1
if isfield(options,'beta') && (options.beta > 0) && (options.beta < 1)
    %�ж������betaֵ�Ƿ�Ϊ�ṹ������options���򣨳�Ա��
    %��beta��ֵӦ������0��1֮��
    beta = options.beta;
end
 
Tag = unique(gnd);     %��ѵ�����������ʸ��
nTag = length(Tag);    %nLabelΪ���ʸ���ĳ��ȣ��������
 
Ww = zeros(nSam,nSam);     %���Ȩ�ؾ���
Wb = ones(nSam,nSam);      %����Ȩ�ؾ���
for idx=1:nTag     %�Ը����������ݽ��з��ദ��
    classIdx = find(gnd==Tag(idx));
    Ww(classIdx,classIdx) = 1;
    Wb(classIdx,classIdx) = 0;
end
 
if k > 0    %��k��ֵ����0ʱ
    D = EuDist2(SampleData,[],0);     % ����ŷ�Ͼ���
    [dump idx] = sort(D,2);           % ��ÿ�н�������
    clear D dump
    idx = idx(:,1:options.k+1);
    G = sparse(repmat([1:nSam]',[options.k+1,1]),...
    idx(:),ones(prod(size(idx)),1),nSam,nSam);    %��ȡ�����еķ�0Ԫ��
    G = max(G,G');
    Ww = Ww.*G;     %�ع�Ȩ�ؾ���
    Wb = Wb.*G;
    clear G
end
 
Db = full(sum(Wb,2));    %�����ϡ�����
Wb = -Wb;
for i=1:size(Wb,1)
    Wb(i,i) = Wb(i,i) + Db(i);     %������������������
end
 
D = full(sum(Ww,2));     %�����ϡ�����
 
if isfield(options,'Regu') && options.Regu     %Regu��Ĭ��ֵΪ1
    options.ReguAlpha = options.ReguAlpha*sum(D)/length(D);
end
 
W = sparse((beta/(1-beta))*Wb+Ww);
clear Wb Ww
 
% ���������������̫����ִ�����´���
if isfield(options,'keepMean') && options.keepMean   %�ж�keepMean������ֵ��
else
    if issparse(SampleData)              %����������������Ϊϡ�����
        SampleData = full(SampleData);   %����������ת���ɷ�ϡ�����
    end
    sampleMean = mean(SampleData);               %�����������ݵ�ƽ��ֵ
    SampleData = (SampleData - repmat(sampleMean,nSam,1));  %�ع�������������
end
 
if ~isfield(options,'Regu') || ~options.Regu     %�ж�Regu������ֵ��
    DToPowerHalf = D.^.5;
    D_mhalf = DToPowerHalf.^-1;
    if nSam < 5000   %���������ݾ��������С��5000����ΪС�������ݣ��򡭡�
        tmpD_mhalf = repmat(D_mhalf,1,nSam);
        W = (tmpD_mhalf.*W).*tmpD_mhalf';
        clear tmpD_mhalf;
    else   %���������������ݹ����򡭡�
        [i_idx,j_idx,v_idx] = find(W);    %������λ��
        v1_idx = zeros(size(v_idx));
        for i=1:length(v_idx)
            v1_idx(i) = v_idx(i)*D_mhalf(i_idx(i))*D_mhalf(j_idx(i));
        end
        W = sparse(i_idx,j_idx,v1_idx);     %ȡ�����Ԫ��
        clear i_idx j_idx v_idx v1_idx
    end
    W = max(W,W');
    SampleData = repmat(DToPowerHalf,1,nCol).*SampleData;%�ع�����
    [eigvector, eigvalue, bSuccess] = OLGE(W, [], options, SampleData);
    %����LGE�����������ά����data������ͼ��Ƕ��
else
    D = sparse(1:nSam,1:nSam,D,nSam,nSam); 
    [eigvector, eigvalue, bSuccess] = OLGE(W, D, options, SampleData);
    %����LGE�����������ά����data������ͼ��Ƕ��
end
 
eigIdx = find(eigvalue < 1e-10);  % ȡ����ֵС��1*10^(-10)����Ӧ������ֵ
eigvalue (eigIdx) = [];          % ������ֵС��1*10^(-10)������ֵ���
eigvector(:,eigIdx) = [];         % ������ֵС��1*10^(-10)����������ȥ��


