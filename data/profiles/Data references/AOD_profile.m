clc
clear
close all;
load AOD_profile.mat
zz=0:0.01:10; % in km
s1=interp1(spring(:,2),spring(:,1),zz,'linear','extrap');
s2=interp1(summer(:,2),summer(:,1),zz,'linear','extrap');
s3=interp1(fall(:,2),fall(:,1),zz,'linear','extrap');
s4=interp1(winter(:,2),winter(:,1),zz,'linear','extrap');
s1(s1<0)=0;
s2(s2<0)=0;
s3(s3<0)=0;
s4(s4<0)=0;
allS=s1/s1(1)+s2/s2(1)+s3/s3(1)+s4/s4(1);
allS=allS'/4;
allS2=(s1+s2+s3+s4)/4;
allS2=allS2'/allS2(1);
Z=zz'*1000; % in meter
data=[Z,allS,s1'/s1(1),s2'/s2(1),s3'/s3(1),s4'/s4(1)];
figure;
plot(Z,allS,Z,allS2,Z,s1'/s1(1),Z,s2'/s2(1),Z,s3'/s3(1),Z,s4'/s4(1));
legend('all','all2','spring','summer','fall','winter');

% seven station altitude
alt=[213,1689,1007,634,98,376,437];
ratio=interp1(data(:,1),data(:,2),alt); % summer profile
mean_AOD=[0.149478,0.076297,0.0771,0.0992,0.158,0.1665,0.1433];
surf_AOD=mean_AOD./ratio;
mean(surf_AOD)
figure;
plot(alt,mean_AOD,'bd');