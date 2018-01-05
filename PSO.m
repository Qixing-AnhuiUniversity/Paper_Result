clear all

global D

T = 500;                              %��������t
N = 20;                              %΢��Ⱥ��ģ
D = 1;                             %΢��Ⱥ��ά��������

Xmax = 3;                         
Xmin = -3;                        %΢��λ�ñ仯��������
Vmax = 0.5;
Vmin = -0.5;                         %΢���ٶȱ仯�������� 


c1 = 2;
c2 = 2;

X = rand(N,D)*(Xmax-Xmin)-Xmax;                  %��[Xmin,Xmax]������������Ǹ�N*Dά����
V = rand(N,D)*(Vmax-Vmin)-Vmax;              %��[Vmin,Vmax]���������Vid(v1d,...,v11d)
tic
                   
for i = 1:N
    Pbestf(i) = Sphere(X(i,:));               
    Pbest(i,:) = X(i,:);                         %��ʼ�������������Ӽ��ϣ����������ǵ���Ӧ�Ⱥ������Ժ�ͱȽ���С�����滻
end
[Gbestf index] = min(Pbestf); %�����Ż�������Ѱ����Сֵ��Ϊȫ�����ŵ���Ӧ�Ⱥ���
Gbest = X(index,:);                            %�ҵ����ȫ�����ŵ�����
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%���¿�ʼ��ѭ��%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%l
F = [];
for t = 1:T                                  %��������1000
    w = 0.9-(0.9-0.4)*t/T;               %Ȩֵ��0.9���α䵽0.4
    %%%%%%%%%%%%%%%%%%����Vid,Xid%%%%%%%%%%%%%%%%%%%%%
    for i = 1:N                                 %N΢��Ⱥ��ģ
          for d = 1:D                             %D΢��Ⱥ��ά��������
            V(i,d) = w*V(i,d)+c1*rand(1,1)*(Pbest(i,d)-X(i,d))+c2*rand(1,1)*(Gbest(d)-X(i,d));%����΢�����ٶ�Vid
            if V(i,d) >Vmax
                V(i,d) = Vmax;
            else if V(i,d) < Vmin
                    V(i,d) = Vmin;
                end
            end                                 %���ٶȵ�������
            X(i,d) = X(i,d)+V(i,d);             %����΢����λ�� Xid    
          if X(i,d) >Xmax
                X(i,d) = Xmax;
            else if X(i,d) < Xmin
                    X(i,d) = Xmin;
                end
            end                                %��΢��λ�õ�������
             
          end 
           %%%%%%%%%%%%%%%��������Xi����Ӧ�Ⱥ���%%%%%%%%%%%%%%%%%
           Xfitness = Sphere(X(i,:));
           %%%%%%%%%%%%%%�ǲ�����Ҫ����Pbest��Gbest����Ҫ��������򱣳�%%%%%%%%%%%%
             if Xfitness <= Pbestf(i)
                Pbest(i,:) = X(i,:);
                Pbestf(i) = Xfitness;
            end                                 %����Pbest
               if Pbestf(i) <= Gbestf
                Gbest = Pbest(i,:);
                Gbestf = Pbestf(i);
                index=i;
            end                                 %����Gbest
     end                                         %����Vid,Xid
     F = [F;Gbestf]; 
 end
 i = [1:1:T];
   plot(i,F)
    toc