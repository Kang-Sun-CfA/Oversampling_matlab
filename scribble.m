validmask = true(60,1644);
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
 
