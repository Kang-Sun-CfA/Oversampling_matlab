function s0 = F_merge_l2g_data(s0,s1)
% merge two l2g structures, s0 and s1. translated from python version,
% popy.F_merg_l2g_data by Kang Sun on 2019/08/25
if isempty(s0)
    s0 = s1;
    return
end
fn0 = fieldnames(s0);
fn1 = fieldnames(s1);
fn = fn0(ismember(fn0,fn1));
for f = fn(:)'
    s0.(f{:}) = cat(1,s0.(f{:}),s1.(f{:}));
end