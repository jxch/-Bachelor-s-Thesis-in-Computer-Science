function [eigvector, eigvalue] = OIsoP(options, data)
% OIsoP: Orthogonal Isometric Projection �����Ⱦ�ӳ���㷨
%             ����:
%               data    - ���ݾ���ÿ����������һ�����ݵ�
%               options - ��������:
%                      NeighborMode -  ָʾ��ι���ͼ��
%                           'KNN'     -  ���ҽ����������ڱ˴˵�k���ڽ�����ʱ���������ڵ�֮�����ñ�Ե��Ĭ��ѡ�
%                           'Supervised'  -  �ල������ѡ��:
%                                       1. k = 0�����ҽ�����������ͬһ����ʱ���������ڵ�֮�����һ���ߡ�
%                                       2. k> 0��ͬһ���������ڵ�֮��ľ���С�������ڵ��diff����ǩӦ�ṩ��ǩ��Ϣ'gnd'��ÿ���������ݵı�ǩʸ������1��2��������n��
%                       k           -   �ھ������� Ĭ�� k = 5;
%                       gnd         -   NeighborMode �µ� 'Supervised'����Ĳ�����ÿ�����ݵ�ı�ǩ��Ϣ����������
%             ���:
%               eigvector - ÿ�ж���Ƕ�뺯��, y = x*eigvector ����x��Ƕ����������������ÿһ�У�
%               eigvalue  - LPP�������������ֵ�� ����С���������



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
% ʹ��K-���ڷ����������ͼG
    G = zeros(nSmp,nSmp);
    for i=1:nLabel
        classIdx = find(options.gnd==Label(i));
        D = EuDist2(data(classIdx,:),[],1);
        G(classIdx,classIdx) = D;
    end
    maxD = max(max(G));
    INF = maxD*INFratio;  % ��Ч���޾���
    
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
            INF = maxD*INFratio;  % ��Ч���޾���
            
            [dump,iidx] = sort(D,2);
            iidx = iidx(:,(2+options.k):end);
            for i=1:nSmp
                D(i,iidx(i,:)) = 0;
            end
            D = max(D,D');
            % ʹ�õϽ�˹�����㷨�������·������D
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

% ʹ��K-���ڷ����������ͼG
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
                    % ʹ�õϽ�˹�����㷨�������·������D
                    D = max(D,D');
                    D = sparse(D);
                    D = dijkstra(D, 1:nSmpClass);
                    G(classIdx,classIdx) = D;
                end
            end
            
            INF = maxD*INFratio;  % ��Ч���޾���

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
% �������������ע�����´���
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
[eigvector, eigvalue] = LGE(TauDg, [], options, data);%����LGE�����������ά����data������ͼ��Ƕ��


eigIdx = find(eigvalue < 1e-3);% ȡ����ֵС��1*10^(-10)����Ӧ������ֵ
eigvalue (eigIdx) = [];% ������ֵС��1*10^(-10)������ֵ���
eigvector(:,eigIdx) = [];% ������ֵС��1*10^(-10)����������ȥ��



