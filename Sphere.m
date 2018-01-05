function f = Sphere(X)  

    Uf = 400;
    Ul = 100;
    s = 0.07;
    sl = 1;
    w = 100*pi;
    wl = 100*pi;
    Rs = 0.087;
    Lls = 0.8e-3;
    Llr = 0.8e-3;
    
    f1 = ((3*Uf^2*X(1)/((s*w)*(Rs+X(1)/s)^2+w^2*(Lls+Llr)^2)-329)/329)^2;
    f2 = ((3*Ul^2*X(1)/((sl*wl)*(Rs+X(1)/sl)^2+wl^2*(Lls+Llr)^2)-186)/186)^2;
    f3 = ((3*Uf^2*X(1)/((2*w)*(Rs+sqrt(Rs^2+w^2*(Lls+Llr)^2)))-398)/398)^2;
    f = f1+f2+f3
   


end
