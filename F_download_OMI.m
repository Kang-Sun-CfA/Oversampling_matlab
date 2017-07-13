function output = F_download_OMI(inp)
% Download NASA OMNO2 data selectively
% The L2 data will be saved under L2dir/yyyy/doy.

% not work for windows OS.
% useful for both OMI and OMPS, just change the BDR file

% Updated by Kang Sun from F_download__subset_OMNO2 on 2017/07/13
% the previous function has problem sorting the downloading/subsetting,
% keeps giving errors when using parfor.

Startdate = inp.Startdate;
Enddate = inp.Enddate;
MinLat = inp.MinLat;
MinLon = inp.MinLon;
MaxLat = inp.MaxLat;
MaxLon = inp.MaxLon;

% if download xml, meta data files
if_download_xml = inp.if_download_xml;

% if download the he5 data
if_download_he5 = inp.if_download_he5;

L2dir = inp.L2dir;
cd(L2dir)

% location of a rough boundary of OMI swath. see /data/tempo1/Shared/kangsun/OMNO2/Important_constant/OMI_BDR.mat
% for an example

swath_BDR_fn = inp.swath_BDR_fn;


url0 = inp.url0;% = 'https://aura.gesdisc.eosdis.nasa.gov/data/Aura_OMI_Level2/OMNO2.003/';

bdrstruct = load(swath_BDR_fn);
left_bdr = bdrstruct.left_bdr;
right_bdr = bdrstruct.right_bdr;

day_array = (datenum(Startdate):1:datenum(Enddate))';
datevec_array = datevec(day_array);
Jan01_array = datevec_array;
Jan01_array(:,2:end) = 0;
doy_array = day_array-datenum(Jan01_array);
year_array = datevec_array(:,1);

nday = length(day_array);
output = cell(nday,1);
parfor iday = 1:nday
    day_dir = [num2str(year_array(iday)),'/',sprintf('%03d',doy_array(iday)),'/'];
    url1 = [url0,day_dir];
    if ~exist(day_dir,'dir')
        mkdir(day_dir)
    end
    cd(day_dir)
    % wget commend to download all xml meta data
    if if_download_xml
        str = ['wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --keep-session-cookies -r -c -nH -nd -np -A xml "',...
            url1,'"'];
        unix(str);
    end
    flist = dir;
    count = 0;
    L2_file_path = cell(1);
    for ixml = 1:length(flist)
        % all xml files, not too small (. and ..), not too large (he5)
        if flist(ixml).bytes > 10000 && flist(ixml).bytes < 1000000
            % open xml and decide if need to download he5 data
            fid = fopen(flist(ixml).name);
            while 1
                s = fgetl(fid);
                if ~ischar(s), break, end;
                if strfind(s,'<EquatorCrossingLongitude>')
                    EqLon = cell2mat(textscan(s,'<EquatorCrossingLongitude>%f</EquatorCrossingLongitude>'));
                    break;
                end
            end
            fclose(fid);
            this_left_bdr = left_bdr;
            this_right_bdr = right_bdr;
            this_left_bdr(1,:) = this_left_bdr(1,:)+EqLon;
            this_right_bdr(1,:) = this_right_bdr(1,:)+EqLon;
            %             plot(this_left_bdr(1,:),this_left_bdr(2,:),this_right_bdr(1,:),this_right_bdr(2,:))
            orbit_ul = interp1(this_left_bdr(2,:),this_left_bdr(1,:),MaxLat);
            orbit_ur = interp1(this_right_bdr(2,:),this_right_bdr(1,:),MaxLat);
            orbit_ll = interp1(this_left_bdr(2,:),this_left_bdr(1,:),MinLat);
            orbit_lr = interp1(this_right_bdr(2,:),this_right_bdr(1,:),MinLat);
            % decide if he5 file worth download
            if (MinLon >= orbit_ul && MinLon <= orbit_ur) ||...
                    (MaxLon >= orbit_ul && MaxLon <= orbit_ur) ||...
                    (MinLon >= orbit_ll && MinLon <= orbit_lr) ||...
                    (MaxLon >= orbit_ll && MaxLon <= orbit_lr) ||...
                    (MinLon <= orbit_ul && MaxLon >= orbit_ur) ||...
                    (MinLon <= orbit_ll && MaxLon >= orbit_lr)
                fn = flist(ixml).name(1:end-4);
                count = count+1;
                L2_file_path{count} = [day_dir,fn];
                if ~exist(fn,'file')
                    if if_download_he5
                        % download the he5 file
                        str = ['wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --keep-session-cookies ',...
                            url1,fn];
                        unix(str);
                    end
                end
            end
        end
    end
    output{iday} = L2_file_path;
    cd(L2dir)
end


