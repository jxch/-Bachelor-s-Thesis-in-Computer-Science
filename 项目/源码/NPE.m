function [eigvector, eigvalue] = NPE(options, data)
% NPE: Neighborhood Preserving Embedding  ���򱣳�Ƕ��
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

if (~exist('options','var'))
   options = [];
end

if ~isfield(options,'NeighborMode')
    options.NeighborMode = 'KNN';
end

if ~isfield(options,'k') 
    options.k = 5;
end



[nSmp,nFea] = size(data);%(�У���)

if options.k >= nSmp
    error('k is too large!');
end

if(options.k > nFea)
    tol=1e-3; % ����Լ��������£��������������ܵ�����
else
    tol=1e-12;
end


% ��һ�����������ͼ��ʹ��K-���ڷ�Ѱ�������ݵ�ŷ�Ͼ��������K�����ڵ㡣
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
% �ڶ�����ȷ��Ȩֵ��
    W = zeros(nSmp,nSmp);
    for ii=1:nSmp
        idx = find(options.gnd==options.gnd(ii));
        idx(find(idx==ii)) = [];
        z = data(idx,:)-repmat(data(ii,:),length(idx),1); % ������ת�Ƶ�ԭ��
        C = z*z';                                   % �ֲ�Э����
        C = C + eye(size(C))*tol*trace(C);                   % ���滯
        tW = C\ones(length(idx),1);                           % solve Cw=1
        tW = tW/sum(tW);                  % enforce sum(w)=1
        W(idx,ii) = tW;
    end
    M = (eye(size(W)) - W);
    M = M*M';
    M = max(M,M');
    M = sparse(M);
else
% ��һ�����������ͼ��ʹ��K-���ڷ�Ѱ�������ݵ�ŷ�Ͼ��������K�����ڵ㡣
    switch lower(options.NeighborMode)
        case {lower('KNN')}

            Distance = EuDist2(data,[],0); %ŷ�Ͼ���
            [sorted,index] = sort(Distance,2);%����
            neighborhood = index(:,2:(1+options.k));%�ҵ�ǰK��
            
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
%ÿ���������ݵı�ǩʸ��
            Label = unique(options.gnd);
            nLabel = length(Label);
            neighborhood = zeros(nSmp,options.k);
            for idx=1:nLabel
                classIdx = find(options.gnd==Label(idx));
                if options.k >= length(classIdx)
                    error('k is too large!');
                end
                Distance = EuDist2(data(classIdx,:),[],0); %ŷ�Ͼ���
                [sorted,index] = sort(Distance,2);%����
                neighborhood(classIdx,:) = classIdx(index(:,2:(1+options.k)));%�ҵ�ǰK��
            end
        otherwise
            error('NeighborMode does not exist!');
    end
% �ڶ�����ȷ��Ȩֵ���ý��ڶԸ������ݵ�����ع���
    W = zeros(options.k,nSmp);
    for ii=1:nSmp
        z = data(neighborhood(ii,:),:)-repmat(data(ii,:),options.k,1); % ������ת�Ƶ�ԭ��
        C = z*z';                                        % �ֲ�Э����
        C = C + eye(size(C))*tol*trace(C);                   % ���滯
        W(:,ii) = C\ones(options.k,1);                           % Cw=1
        W(:,ii) = W(:,ii)/sum(W(:,ii));                  % sum(w)=1
    end

    M = sparse(1:nSmp,1:nSmp,ones(1,nSmp),nSmp,nSmp,4*options.k*nSmp);%ϡ��
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


% ����������������ӳ�䡣
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

M = -M;
for i=1:size(M,1)
    M(i,i) = M(i,i) + 1;
end



[eigvector, eigvalue] = LGE(M, [], options, data);%����LGE�����������ά����data������ͼ��Ƕ��


eigIdx = find(eigvalue < 1e-10);% ȡ����ֵС��1*10^(-10)����Ӧ������ֵ
eigvalue (eigIdx) = [];% ������ֵС��1*10^(-10)������ֵ���
eigvector(:,eigIdx) = [];% ������ֵС��1*10^(-10)����������ȥ��




