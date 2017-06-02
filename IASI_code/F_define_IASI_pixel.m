function [u, v, t] = F_define_IASI_pixel(lat,ifov,uuu4,vvv4,ttt4)
% calculate ellipse pixel with look up table
ifov = round(ifov);
u = interp2(-90:90,1:120,uuu4',lat,ifov);
v = interp2(-90:90,1:120,vvv4',lat,ifov);
t = interp2(-90:90,1:120,ttt4',lat,ifov);