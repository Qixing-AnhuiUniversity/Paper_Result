clear all

global D

T = 500;                             
N = 20;                             
D = 1;                             

Xmax = 3;                         
Xmin = -3;                        
Vmax = 0.5;
Vmin = -0.5;                         


c1 = 2;
c2 = 2;

X = rand(N,D)*(Xmax-Xmin)-Xmax;                  
V = rand(N,D)*(Vmax-Vmin)-Vmax;              
tic
                   
for i = 1:N
    Pbestf(i) = Sphere(X(i,:));               
    Pbest(i,:) = X(i,:);                        
end
[Gbestf index] = min(Pbestf); 
Gbest = X(index,:);                           

F = [];
for t = 1:T                                 
    w = 0.9-(0.9-0.4)*t/T;               
    
    for i = 1:N                                 
          for d = 1:D                             
            V(i,d) = w*V(i,d)+c1*rand(1,1)*(Pbest(i,d)-X(i,d))+c2*rand(1,1)*(Gbest(d)-X(i,d));
            if V(i,d) >Vmax
                V(i,d) = Vmax;
            else if V(i,d) < Vmin
                    V(i,d) = Vmin;
                end
            end                                 
            X(i,d) = X(i,d)+V(i,d);                
          if X(i,d) >Xmax
                X(i,d) = Xmax;
            else if X(i,d) < Xmin
                    X(i,d) = Xmin;
                end
            end                                
             
          end 
           
           Xfitness = Sphere(X(i,:));
          
             if Xfitness <= Pbestf(i)
                Pbest(i,:) = X(i,:);
                Pbestf(i) = Xfitness;
            end                                 
               if Pbestf(i) <= Gbestf
                Gbest = Pbest(i,:);
                Gbestf = Pbestf(i);
                index=i;
            end                                 
     end                                         
     F = [F;Gbestf]; 
 end
 i = [1:1:T];
   plot(i,F)
    toc
