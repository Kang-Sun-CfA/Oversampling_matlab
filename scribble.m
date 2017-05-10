validmask = false(60,1644);
validmask(60,300) = true;
    Lat_lowerleft = datavar.PixelCornerLatitudes.data(1:end-1,1:end-1);
    Lat_lowerright = datavar.PixelCornerLatitudes.data(2:end,1:end-1);
    Lat_upperleft = datavar.PixelCornerLatitudes.data(1:end-1,2:end);
    Lat_upperright = datavar.PixelCornerLatitudes.data(2:end,2:end);
    
    Lon_lowerleft = datavar.PixelCornerLongitudes.data(1:end-1,1:end-1);
    Lon_lowerright = datavar.PixelCornerLongitudes.data(2:end,1:end-1);
    Lon_upperleft = datavar.PixelCornerLongitudes.data(1:end-1,2:end);
    Lon_upperright = datavar.PixelCornerLongitudes.data(2:end,2:end);
    
    tempLatC = datavar.Latitude(validmask);
    tempLonC = datavar.Longitude(validmask);
    
    tempLat_lowerleft = Lat_lowerleft(validmask);
    tempLat_lowerright = Lat_lowerright(validmask);
    tempLat_upperleft = Lat_upperleft(validmask);
    tempLat_upperright = Lat_upperright(validmask);
    
    tempLon_lowerleft = Lon_lowerleft(validmask);
    tempLon_lowerright = Lon_lowerright(validmask);
    tempLon_upperleft = Lon_upperleft(validmask);
    tempLon_upperright = Lon_upperright(validmask);
    %     % plot the corners to see if it's alright
    %     plot(tempLonC,tempLatC,'.k',tempLon_lowerleft,tempLat_lowerleft,'.'...
    %         ,tempLon_lowerright,tempLat_lowerright,'o',tempLon_upperleft,tempLat_upperleft,'v'...
    %         ,tempLon_upperright,tempLat_upperright,'*')
    
    tempdata = [tempLat_lowerleft(:),tempLat_upperleft(:),...
        tempLat_upperright(:),tempLat_lowerright(:),tempLatC(:),...
        tempLon_lowerleft(:),tempLon_upperleft(:),...
        tempLon_upperright(:),tempLon_lowerright(:),tempLonC(:),...
        ];
%% 

Lon_r = tempdata(6:9);
Lat_r = tempdata(1:4);
Lon_c = tempdata(10);
Lat_c = tempdata(5);
Res = 0.05;
pixel.nv = length(Lon_r);
pixel.vList = [(Lon_r(:)-Lon_left)/Res,(Lat_r(:)-Lat_low)/Res];

Lon_left = floor(min(Lon_r));
Lon_right = ceil(max(Lon_r));
Lat_low = floor(min(Lat_r));
Lat_up = ceil(max(Lat_r));



[pout,npout] = F_HcakeCut(pixel);
close all;
subplot(1,2,1)
hold on
for i = 1:npout
    F_plot_polygon(pout(i));
end
[pout,npout] = F_VcakeCut(pixel);
axis equal
subplot(1,2,2)
hold on
for i = 1:npout
    F_plot_polygon(pout(i));
end
axis equal
% test clockwise
figure
hold on
plot(Lon_c,Lat_c,'ok')
for i = 1:length(Lon_r)
    plot(Lon_r(i),Lat_r(i),'*');
end
hold off
axis equal