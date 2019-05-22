function orthogonalMatrix=Schmidt_orthogonalization(originalMatrix)
% 施密特正交化

[m,n] = size(originalMatrix);

if(m<n)
    error('行小于列，无法计算，请转置后重新输入');
    return;
end

orthogonalMatrix=zeros(m,n);
%正交化
orthogonalMatrix(:,1)=originalMatrix(:,1);
for i=2:n
    for j=1:i-1
        orthogonalMatrix(:,i)=orthogonalMatrix(:,i)-dot(originalMatrix(:,i),orthogonalMatrix(:,j))/dot(orthogonalMatrix(:,j),orthogonalMatrix(:,j))*orthogonalMatrix(:,j);
    end
    orthogonalMatrix(:,i)=orthogonalMatrix(:,i)+originalMatrix(:,i);
end

%单位化
% for k=1:n
%     orthogonalMatrix(:,k)=orthogonalMatrix(:,k)/norm(orthogonalMatrix(:,k));
% end