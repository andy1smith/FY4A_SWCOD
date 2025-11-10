% read copy-pasted data from contnm.f90
function coeff=readContmData(fileID)
C = textscan(fileID,'%s','Delimiter',{',','&','/'});
N=length(C{1,1});
for i=1:N
    temp{i,1}=str2num(C{1,1}{i});
end
try
coeff=cell2mat(temp); % in unit of cm2/mol (cm)-1
catch
    coeff=temp;
end
end