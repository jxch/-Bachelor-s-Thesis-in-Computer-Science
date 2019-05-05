clear all
clc
close all
start=clock;
sample_class=1:40;%�������
sample_classnum=size(sample_class,2);%���������
fprintf('\n\n*******************************����LSDA�㷨������ʶ��********************************\n\n');
fprintf('�������п�ʼ....................\n\n');

for train_samplesize=3:9;
    train=1:train_samplesize;%ÿ��ѵ������
    test=train_samplesize+1:10;%ÿ���������
    
    train_num=size(train,2);%ÿ��ѵ��������
    test_num=size(test,2);%ÿ�����������
    
    address=[pwd '\ORL\s'];
    %��ȡѵ������
    allsamples=readsample(address,sample_class,train);

    %%%%%%%%%%%%%%%%%%%%%%%%%%  ʹ��LSDA���н�ά %%%%%%%%%%%%%%%%%%%%%%%
    num=train_num;
    gnd = [1*ones(1,num),2*ones(1,num),3*ones(1,num),4*ones(1,num),...
        5*ones(1,num),6*ones(1,num),7*ones(1,num),8*ones(1,num),...
        9*ones(1,num),10*ones(1,num),11*ones(1,num),12*ones(1,num),...
        13*ones(1,num),14*ones(1,num),15*ones(1,num),16*ones(1,num),...
        17*ones(1,num),18*ones(1,num),19*ones(1,num),20*ones(1,num),...
        21*ones(1,num),22*ones(1,num),23*ones(1,num),24*ones(1,num),...
        25*ones(1,num),26*ones(1,num),27*ones(1,num),28*ones(1,num),...
        29*ones(1,num),30*ones(1,num),31*ones(1,num),32*ones(1,num),...
        33*ones(1,num),34*ones(1,num),35*ones(1,num),36*ones(1,num),...
        37*ones(1,num),38*ones(1,num),39*ones(1,num),40*ones(1,num)]; 
    options = [];
    options.k = 5;
    [eigvector, eigvalue] = LSDA(gnd, options, allsamples);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %[m6,n6]=size(base)
    %[m9,n9]=size(newsample)
    
    newsample=allsamples*eigvector;  %newsample��ʾ��eigvectorӳ���»�õ�������ʾ
    %[m9,n9]=size(newsample)
    
    %����Sw,Sb
    [sw sb]=computswb(newsample,sample_classnum,train_num);
    
    %��ȡ��������
    testsample=readsample(address,sample_class,test); 
    %[m5,n5]=size(testsample)   
    best_acc=0;%����ʶ����
    %Ѱ�����ͶӰά��
    for temp_dimension=1:1:length(sw)
        vsort1=projectto(sw,sb,temp_dimension);
        %[m7,n7]=size(vsort1)
        %ѵ�������Ͳ��������ֱ�ͶӰ
        tstsample=testsample*eigvector*vsort1;   
        trainsample=newsample*vsort1;

        %����ʶ����
        accuracy=computaccu(tstsample,test_num,trainsample,train_num);
        if accuracy>best_acc
            best_dimension=temp_dimension;%�������ͶӰά��
            best_acc=accuracy;
        end
    end
    %---------------------------------�����ʾ----------------------------------
    fprintf('ÿ��ѵ��������Ϊ��%d\n',train_samplesize);
    fprintf('���ͶӰά��Ϊ��%d\n',best_dimension);
    fprintf('LSDA��ʶ����Ϊ��%.2f%%\n',best_acc*100);
    fprintf('��������ʱ��Ϊ��%3.2fs\n\n',etime(clock,start));
end
fprintf('�������н���....................\n\n');