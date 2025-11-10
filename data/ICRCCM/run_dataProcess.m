%% process data from ICRCCM results
close all
clear
clc
fileID=fopen('aer.c27.spectral'); % CKD coeffs
data=readICRCCMData(fileID);
csvwrite('ICRCCM_27_spectral.csv',data);
%%
trapz(data(:,1),data(:,3))/5
sum(data(:,3))