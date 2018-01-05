for t = 1:1:10000
ad1(t) = disddt(:,:,t+1000);
ad2(t) = isd(:,:,t+1000);
ad3(t) = -slip(:,:,t+1000)*isq(:,:,t+1000);
ad4(t) = -w(:,:,t+1000)*isq(:,:,t+1000);
ad5(t) = -(dvsddt(:,:,t+1000)-slip(:,:,t+1000)*vq(:,:,t+1000)-1/w(:,:,t+1000)*dwmddt(:,:,t+1000)*vd(:,:,t+1000));
ad6(t) = -vd(:,:,t+1000);
aq1(t) = disqdt(:,:,t+1000);
aq2(t) = isq(:,:,t+1000);
aq3(t) = slip(:,:,t+1000)*isd(:,:,t+1000);
aq4(t) = w(:,:,t+1000)*isd(:,:,t+1000);
aq5(t) = -(dvsqdt(:,:,t+1000)+slip(:,:,t+1000)*vd(:,:,t+1000)-1/w(:,:,t+1000)*dwmddt(:,:,t+1000)*vq(:,:,t+1000));
aq6(t) = -vq(:,:,t+1000);
bd(t)=-disd2dt(:,:,t+1000)+(w(:,:,t+1000)+slip(:,:,t+1000))*disqdt(:,:,t+1000)+w(:,:,t+1000)*(slip(:,:,t+1000))*isd(:,:,t+1000)-dwmddt(:,:,t+1000)*isq(:,:,t+1000);
bq(t)=-disq2dt(:,:,t+1000)-(w(:,:,t+1000)+slip(:,:,t+1000))*disddt(:,:,t+1000)+w(:,:,t+1000)*(slip(:,:,t+1000))*isq(:,:,t+1000)+dwmddt(:,:,t+1000)*isd(:,:,t+1000);
end
csvwrite('ad1.csv',ad1')
csvwrite('ad2.csv',ad2')
csvwrite('ad3.csv',ad3')
csvwrite('ad4.csv',ad4')
csvwrite('ad5.csv',ad5')
csvwrite('ad6.csv',ad6')
csvwrite('aq1.csv',aq1')
csvwrite('aq2.csv',aq2')
csvwrite('aq3.csv',aq3')
csvwrite('aq4.csv',aq4')
csvwrite('aq5.csv',aq5')
csvwrite('aq6.csv',aq6')
csvwrite('bd.csv',bd')
csvwrite('bq.csv',bq')