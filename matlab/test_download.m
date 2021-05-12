load d:\research_CfA\OMNO2\2017output_download.mat
a = [];
for iday = 1:length(output_download.exist)
    a = cat(2,a,cell2mat(output_download.exist{iday}));
end
plot(a)