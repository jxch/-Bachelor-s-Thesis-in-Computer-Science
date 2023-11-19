function orthogonalMatrix=Schmidt_orthogonalization(originalMatrix)
% ʩ����������

[m,n] = size(originalMatrix);

if(m<n)
    error('��С���У��޷����㣬��ת�ú���������');
    return;
end

orthogonalMatrix=zeros(m,n);
%������
orthogonalMatrix(:,1)=originalMatrix(:,1);
for i=2:n
    for j=1:i-1
        orthogonalMatrix(:,i)=orthogonalMatrix(:,i)-dot(originalMatrix(:,i),orthogonalMatrix(:,j))/dot(orthogonalMatrix(:,j),orthogonalMatrix(:,j))*orthogonalMatrix(:,j);
    end
    orthogonalMatrix(:,i)=orthogonalMatrix(:,i)+originalMatrix(:,i);
end

%��λ��
% for k=1:n
%     orthogonalMatrix(:,k)=orthogonalMatrix(:,k)/norm(orthogonalMatrix(:,k));
% end