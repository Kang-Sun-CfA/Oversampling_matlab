clear
regrid_year = [2008 2009 2014 2015];
L2gdir = '/data/tempo1/Shared/kangsun/OMHCHO/L2g/'; % intermediate data, or L2g
L3dir = '/data/tempo1/Shared/kangsun/OMHCHO/L3/'; % regridded data, or L3
%% regrid subsetted L2 data (or L2g data) into L3 data, save L3 monthly
for iyear = regrid_year
    L2g_fn = ['CONUS_',num2str(iyear),'.mat'];
    load([L2gdir,L2g_fn],'inp_subset','output_subset')
    inp_regrid = [];
    % Resolution of oversampled L3 data?
    inp_regrid.Res = 0.1; % in degree
    % CONUS
    inp_regrid.MinLon = -130;
    inp_regrid.MaxLon = -63;
    inp_regrid.MinLat = 25;
    inp_regrid.MaxLat = 50;
    inp_regrid.MaxCF = 0.3;
    inp_regrid.MaxSZA = 60;
    inp_regrid.vcdname = 'colhcho';
    inp_regrid.vcderrorname = 'colhchoerror';
    inp_regrid.if_parallel = true;
    
    end_date_month = [31 28 31 30 31 30 31 31 30 31 30 31];
    if iyear == 2004 || iyear == 2008 || iyear == 2016
        end_date_month = [31 29 31 30 31 30 31 31 30 31 30 31];
    end
    for imonth = 1:12
        inp_regrid.Startdate = [iyear imonth 1];
        inp_regrid.Enddate = [iyear imonth end_date_month(imonth)];
        output_regrid = F_regrid_OMI(inp_regrid,output_subset);
        L3_fn = ['CONUS_',num2str(iyear),'_',num2str(imonth),'.mat'];
        save([L3dir,L3_fn],'inp_regrid','output_regrid')
    end
end
%%
L2gdir = '/data/tempo1/Shared/kangsun/OMNO2/L2g/'; % intermediate data, or L2g
L3dir = '/data/tempo1/Shared/kangsun/OMNO2/L3/'; % regridded data, or L3
%% regrid subsetted L2 data (or L2g data) into L3 data, save L3 monthly
for iyear = regrid_year
    L2g_fn = ['CONUS_',num2str(iyear),'.mat'];
    load([L2gdir,L2g_fn],'inp_subset','output_subset')
    inp_regrid = [];
    % Resolution of oversampled L3 data?
    inp_regrid.Res = 0.1; % in degree
    % CONUS
    inp_regrid.MinLon = -130;
    inp_regrid.MaxLon = -63;
    inp_regrid.MinLat = 25;
    inp_regrid.MaxLat = 50;
    inp_regrid.MaxCF = 0.3;
    inp_regrid.MaxSZA = 60;
    inp_regrid.vcdname = 'colno2';
    inp_regrid.vcderrorname = 'colno2error';
    inp_regrid.if_parallel = true;
    
    end_date_month = [31 28 31 30 31 30 31 31 30 31 30 31];
    if iyear == 2004 || iyear == 2008 || iyear == 2016
        end_date_month = [31 29 31 30 31 30 31 31 30 31 30 31];
    end
    for imonth = 1:12
        inp_regrid.Startdate = [iyear imonth 1];
        inp_regrid.Enddate = [iyear imonth end_date_month(imonth)];
        output_regrid = F_regrid_OMI(inp_regrid,output_subset);
        L3_fn = ['CONUS_',num2str(iyear),'_',num2str(imonth),'.mat'];
        save([L3dir,L3_fn],'inp_regrid','output_regrid')
    end
end
