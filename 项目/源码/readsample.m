function sample=readsample(address,classnum,num)
%�������������ȡ������
%���룺address����Ҫ��ȡ�������ĵ�ַ,classnum����Ҫ�������������,num��ÿ���������
%���Ϊ��������
allsamples=[];
image=imread([pwd '\ORL\s1_1.bmp']);%�����һ��ͼ��
[rows cols]=size(image);%���ͼ�������������
for i=classnum
    for j=num
        a=imread(strcat(address,num2str(i),'_',num2str(j),'.bmp'));
        b=a(1:rows*cols);
        b=double(b);
        allsamples=[allsamples;b];
    end
end
sample=allsamples;