function F_save_gesdisc_url(inp)
% matlab function to parse monthly s5p downloading url files from nasa ges
% disc. written by Kang Sun on 2019/11/13

% % testing
% clear;clc
% inp = [];
% inp.molecule_name = 'NO2';
% inp.raw_txt_flist = {'~/TROPOMI/downloads/s5pno2_tempo.txt','~/TROPOMI/downloads/s5pno2_tempo_HiR.txt'};
% inp.download_dir = '~/TROPOMI/downloads/tempo_no2/';
% inp.if_append = true;

if ~exist(inp.download_dir,'dir')
    mkdir(inp.download_dir);
end
existing_url = dir([inp.download_dir,inp.molecule_name,'_url_*.txt']);
for ifile = 1:length(existing_url)
    delete([inp.download_dir,existing_url(ifile).name])
end
pattern_str = '_L2________';
pattern_str(6:6+length(inp.molecule_name)-1) = inp.molecule_name;
for raw_txt_fn = inp.raw_txt_flist
    raw_fn = raw_txt_fn{:};
    fid = fopen(raw_fn);
    C = cell(2,1);
    count = 0;
    while 1
        tline = fgetl(fid);
        if ~ischar(tline)
            break;
        end
        if contains(tline,'gesdisc.eosdis.nasa.gov')
            count = count+1;
            C{count} = tline;
        end
    end
    fclose(fid);
    
    nfile = length(C);
    file_datevec = nan(nfile,6);
    for ifile = 1:nfile
        k = strfind(C{ifile},pattern_str);
        file_datevec(ifile,:) = datevec(C{ifile}((k(2)+length(pattern_str)+1):(k(2)+length(pattern_str)+15)),'yyyymmddTHHMMSS');
    end
    for y = unique(file_datevec(:,1))'
        for m = unique(file_datevec(:,2))'
            int = file_datevec(:,1) == y & file_datevec(:,2) == m;
            if sum(int) > 0
                if inp.if_append
                    fid = fopen([inp.download_dir,inp.molecule_name,'_url_',num2str(y),'_',num2str(m,'%02d'),'.txt'],'A');
                else
                    fid = fopen([inp.download_dir,inp.molecule_name,'_url_',num2str(y),'_',num2str(m,'%02d'),'.txt'],'w');
                end
                CT = C(int);
                fprintf(fid,'%s\n', CT{:});
                fclose(fid);
            end
        end
    end
    
end
