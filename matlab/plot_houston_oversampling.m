% directory for plot
plotdir = '/data/wdocs/kangsun/www-docs/files/';

% matlab code dir
codedir = '/home/kangsun/OMI/Oversampling_matlab/';

addpath('/home/kangsun/matlab functions/export_fig/')

cd(codedir)
%% load fortran output
fid = fopen('/data/tempo1/Shared/kangsun/Oversampling/RegridPixels/output/Res_0.02_Lat_28_34_Lon_-99.5_-92.5_Year_200505_200808.dat');
C = cell2mat(textscan(fid,'%f%f%f%f%f%f','delimiter',' ','multipledelimsasone',1));
fclose(fid);
[nlat,Ilat] = max(C(:,1));
max_grid_lat = C(Ilat,3);
min_grid_lat = max_grid_lat-(nlat-1)*Res;
latgrid = (1:nlat)*Res+min_grid_lat-Res;

[nlon,Ilon] = max(C(:,2));
max_grid_lon = C(Ilon,4);
min_grid_lon = max_grid_lon-(nlon-1)*Res;
longrid = (1:nlon)*Res+min_grid_lon-Res;

value = nan(nlat,nlon);
nave = value;
for i = 1:size(C,1)
    value(C(i,1),C(i,2)) = C(i,5);
    nave(C(i,1),C(i,2)) = C(i,6);
end
%%
load('L3plot.mat')
statelist = [8, 55, 32, 24 46];
S         = shaperead('/data/tempo1/Shared/kangsun/run_WRF/shapefiles/cb_2015_us_state_500k/cb_2015_us_state_500k.shp');
% Slake     = shaperead('/data/tempo1/Shared/kangsun/run_WRF/shapefiles/ne_10m_lakes_north_america/ne_10m_lakes_north_america.shp');
if ~exist('lakelist','var')
    Llake     = shaperead('/data/tempo1/Shared/kangsun/run_WRF/shapefiles/ne_10m_lakes/ne_10m_lakes.shp');
    lakelist = [];
    left = -82;right = -74;down = 39.5;up = 45;
    for i = 1:length(Llake)
        tmp = Llake(i).BoundingBox;
        if ((tmp(1,1) > left && tmp(1,1) < right) || ...
                (tmp(2,1) > left && tmp(2,1) < right)) && ...
                ((tmp(2,1) > down && tmp(2,1) < up) || ...
                (tmp(2,2) > down && tmp(2,2) < up))
            lakelist = [lakelist i];
        end
    end
end
close all
figure('color','w','unit','inch','position',[0 1 10 9])
set(0,'defaultaxesfontsize',12)
% h = scatter(C(:,4),C(:,3),[],C(:,5));
textsr = {'Tessellation fortran','Tessellation matlab','m = 4, n = 2',...
    'm = 2, n = 2'};
for i = 1:4
    ax = subplot(2,2,i);
    pos = get(ax,'position');
    set(ax,'position',[pos(1)-0.05 pos(2)+0.02 pos(3)*1.2 pos(4)*1.15])
    if i == 1
        h = pcolor(longrid,latgrid,value/1e16);set(h,'edgecolor','none')
    else
        plotmat = plotcell{i-1}/1e16;
        % imagesc(plotmat)
        h = pcolor(double(Lon_grid),double(Lat_grid),double(plotmat));set(h,'edgecolor','none')
    end
    % hc = colorbar;
    ylim([28 34]);xlim([-99.5 -92.5])
    text(-93, 28.2,textsr{i},'color','w','fontsize',13,'horizontalalignment','right')
%     axis equal
    axis off
    caxis([.3 1.5])
    
    hold on
    for istate = 1:length(S)
        plot(S(istate).X,S(istate).Y,'color','w')
    end
    
    for ilake = lakelist
        plot(Llake(ilake).X,Llake(ilake).Y,'color','w')
    end
end
hc = colorbar('north');
set(hc,'position',[0.35 0.09 0.3 0.02])
set(get(hc,'xlabel'),'string','HCHO column [10^{16} molecules cm^{-2}]')
%%
export_fig([plotdir,'L3_compare.png'],'-r200')