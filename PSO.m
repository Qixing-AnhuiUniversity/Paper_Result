clear all

global D

T = 500;                              %迭代次数t
N = 20;                              %微粒群规模
D = 1;                             %微粒群的维数即长度

Xmax = 3;                         
Xmin = -3;                        %微粒位置变化的上下限
Vmax = 0.5;
Vmin = -0.5;                         %微粒速度变化的上下限 


c1 = 2;
c2 = 2;

X = rand(N,D)*(Xmax-Xmin)-Xmax;                  %在[Xmin,Xmax]内随机产生，是个N*D维矩阵
V = rand(N,D)*(Vmax-Vmin)-Vmax;              %在[Vmin,Vmax]内随机产生Vid(v1d,...,v11d)
tic
                   
for i = 1:N
    Pbestf(i) = Sphere(X(i,:));               
    Pbest(i,:) = X(i,:);                         %初始化个体最优粒子集合，并计算它们的适应度函数，以后就比较若小于则替换
end
[Gbestf index] = min(Pbestf); %在最优化个体中寻找最小值作为全局最优的适应度函数
Gbest = X(index,:);                            %找到这个全局最优的粒子
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%以下开始主循环%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%l
F = [];
for t = 1:T                                  %迭代次数1000
    w = 0.9-(0.9-0.4)*t/T;               %权值由0.9线形变到0.4
    %%%%%%%%%%%%%%%%%%计算Vid,Xid%%%%%%%%%%%%%%%%%%%%%
    for i = 1:N                                 %N微粒群规模
          for d = 1:D                             %D微粒群的维数即长度
            V(i,d) = w*V(i,d)+c1*rand(1,1)*(Pbest(i,d)-X(i,d))+c2*rand(1,1)*(Gbest(d)-X(i,d));%计算微粒的速度Vid
            if V(i,d) >Vmax
                V(i,d) = Vmax;
            else if V(i,d) < Vmin
                    V(i,d) = Vmin;
                end
            end                                 %设速度的上下限
            X(i,d) = X(i,d)+V(i,d);             %计算微粒的位置 Xid    
          if X(i,d) >Xmax
                X(i,d) = Xmax;
            else if X(i,d) < Xmin
                    X(i,d) = Xmin;
                end
            end                                %设微粒位置的上下限
             
          end 
           %%%%%%%%%%%%%%%计算粒子Xi的适应度函数%%%%%%%%%%%%%%%%%
           Xfitness = Sphere(X(i,:));
           %%%%%%%%%%%%%%是不是需要更换Pbest，Gbest，需要则更，否则保持%%%%%%%%%%%%
             if Xfitness <= Pbestf(i)
                Pbest(i,:) = X(i,:);
                Pbestf(i) = Xfitness;
            end                                 %更新Pbest
               if Pbestf(i) <= Gbestf
                Gbest = Pbest(i,:);
                Gbestf = Pbestf(i);
                index=i;
            end                                 %更新Gbest
     end                                         %计算Vid,Xid
     F = [F;Gbestf]; 
 end
 i = [1:1:T];
   plot(i,F)
    toc