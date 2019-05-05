clear all
clc
close all
start=clock;
sample_class=1:40;%样本类别
sample_classnum=size(sample_class,2);%样本类别数
fprintf('\n\n*******************************基于LSDA算法的人脸识别********************************\n\n');
fprintf('程序运行开始....................\n\n');

for train_samplesize=3:9;
    train=1:train_samplesize;%每类训练样本
    test=train_samplesize+1:10;%每类测试样本
    
    train_num=size(train,2);%每类训练样本数
    test_num=size(test,2);%每类测试样本数
    
    address=[pwd '\ORL\s'];
    %读取训练样本
    allsamples=readsample(address,sample_class,train);

    %%%%%%%%%%%%%%%%%%%%%%%%%%  使用LSDA进行降维 %%%%%%%%%%%%%%%%%%%%%%%
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
    
    newsample=allsamples*eigvector;  %newsample表示在eigvector映射下获得的样本表示
    %[m9,n9]=size(newsample)
    
    %计算Sw,Sb
    [sw sb]=computswb(newsample,sample_classnum,train_num);
    
    %读取测试样本
    testsample=readsample(address,sample_class,test); 
    %[m5,n5]=size(testsample)   
    best_acc=0;%最优识别率
    %寻找最佳投影维数
    for temp_dimension=1:1:length(sw)
        vsort1=projectto(sw,sb,temp_dimension);
        %[m7,n7]=size(vsort1)
        %训练样本和测试样本分别投影
        tstsample=testsample*eigvector*vsort1;   
        trainsample=newsample*vsort1;

        %计算识别率
        accuracy=computaccu(tstsample,test_num,trainsample,train_num);
        if accuracy>best_acc
            best_dimension=temp_dimension;%保存最佳投影维数
            best_acc=accuracy;
        end
    end
    %---------------------------------输出显示----------------------------------
    fprintf('每类训练样本数为：%d\n',train_samplesize);
    fprintf('最佳投影维数为：%d\n',best_dimension);
    fprintf('LSDA的识别率为：%.2f%%\n',best_acc*100);
    fprintf('程序运行时间为：%3.2fs\n\n',etime(clock,start));
end
fprintf('程序运行结束....................\n\n');