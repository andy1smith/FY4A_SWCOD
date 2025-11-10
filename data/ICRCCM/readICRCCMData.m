% read copy-pasted data from contnm.f90
function B=readICRCCMData(fileID)
C = textscan(fileID,'%s','Delimiter',{',','&','/'});
data=C{1,1};
N=length(data);
for i=1:N
    A{i,1}=str2num(data{i});
end
nu_c=A(1:53:end,1);
flux_c=A(2:53:end,1);
nu=cell2mat(nu_c);
flux=cell2mat(flux_c);
B=[nu(:,1),flux];
end