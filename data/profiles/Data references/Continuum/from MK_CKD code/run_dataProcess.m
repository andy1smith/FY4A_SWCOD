%% process data for self continuum coefficients--H2O
% Last modified 07/23/2018 based on MT_CKD_3.2
close all
clear
clc
fileID=fopen('raw_selfContm_296.txt'); % CKD coeffs
coeff_296=readContmData(fileID);
fileID=fopen('raw_selfContm_260.txt'); % CKD coeffs
coeff_260=readContmData(fileID);
nu=[-20:10:20000]';

%---------get correction factor--------------
fileID=fopen('raw_selfContm_sfac1.txt'); % correction factor
sfac1=readContmData(fileID);
%fileID=fopen('raw_selfContm_sfac2.txt'); % correction factor
%sfac2=readContmData(fileID);

sfac=nu*0+1; %initialize to 1 and add tabulate value
ind1=find(nu==820);
ind2=find(nu==960);
%ind3=find(nu==2000);
%ind4=find(nu==3190);
sfac(ind1:ind2)=sfac1;
%sfac(ind3:ind4)=sfac2;

f1=0.25;
f1_rhu=0.08;
beta=350;
beta_rhu=40;
n_s=6;
ratio1=1+f1./(1+(nu./beta).^n_s);% added on 7/23/18
ratio2=1+f1_rhu./(1+(nu./beta_rhu).^n_s);
sfac=sfac.*ratio1.*ratio2;

plot(nu,log10(coeff_296),'b');
hold on;
plot(nu,log10(coeff_260),'r');
%ratio1=1+f1./(1+(nu./beta).^n_s);
%plot(nu,log10(coeff.*sfac./ratio.*ratio1),'k-');
xlim([0,20000]);

csvwrite('selfContm.csv',[nu,coeff_296,coeff_260,sfac]);
%% process data for foreign continuum coefficients -- H2O
close all
clear
clc
fileID=fopen('raw_frgnContm_296.txt'); % CKD coeffs
coeff_296=readContmData(fileID);
nu=[-20:10:20000]';

%---------get correction factor--------------
fileID=fopen('raw_frgnContm_fscal.txt'); % correction factor
fscal1=readContmData(fileID);
%---compute correction factor
f0=0.06; v0f1=255.67; hwsq1=240^2;
beta1=57.83; c1=-0.42; N1=8;
beta2=630; c2=0.3; N2=8;

vdelsq1=(nu-v0f1).^2;
vdelmsq1=(nu+v0f1).^2;
vf1=((nu-v0f1)./beta1).^N1;
vmf1=((nu+v0f1)./beta1).^N1;
vf2=(nu./beta2).^N2;
temp1=hwsq1./(vdelsq1+hwsq1+vf1);
temp2=hwsq1./(vdelmsq1+hwsq1+vmf1);
fscal=1+(f0+c1.*(temp1+temp2))./(1+c2.*vf2);
ind1=find(nu==-20);
ind2=find(nu==600);
fscal(ind1:ind2)=fscal1;

plot(nu,log10(coeff_296),'b');
hold on;
plot(nu,log10(coeff_296.*fscal),'r');
%ratio1=1+f1./(1+(nu./beta).^n_s);
%plot(nu,log10(coeff.*sfac./ratio.*ratio1),'k-');
xlim([0,20000]);
csvwrite('frgnContm.csv',[nu,coeff_296,fscal]);

%% process data for self continuum coefficients--CO2
close all
clear
clc
nu=-4:2:10000;
coeff=nan(length(nu),4);
coeff(:,1)=nu;
fileID=fopen('raw_BFCO2.txt'); % CKD coeffs
coeff(:,2)=readContmData(fileID);
fileID=fopen('raw_XFACCO2.txt'); % correction factor for 2000-3000 cm-1, dnu=2 cm-1
cfac=readContmData(fileID);
coeff(:,3)=interp1([2000:2:3000-2]',cfac,nu,'linear',1);
fileID=fopen('raw_tdep_bandhead.txt'); % correction factor for 2000-3000 cm-1, dnu=2 cm-1
tdep=readContmData(fileID);
coeff(:,4)=interp1([1196:1220]',tdep,nu,'linear',0);

csvwrite('frgnContm_CO2.csv',[coeff]);

%% process data for shortwave continuum -- O3 -- separate as 3 bands on July 13th
close all
clear
clc
% Band 1
nu_1=[8920:5:24665]';
coeff_1=repmat(nu_1*0,[1,3]);
files={'raw_O3CH_X.txt','raw_O3CH_Y.txt','raw_O3CH_Z.txt'};
for i=1:length(files)
    fileID=fopen(files{i}); % CKD coeffs
    temp=readContmData(fileID);
    coeff_1(:,i)=temp./nu_1; % remove radiation field
end
csvwrite('Contm_O3_b1.csv',[nu_1,coeff_1]);
% testing
c0=coeff_1(:,1);
c1=coeff_1(:,2);
c2=coeff_1(:,3);
DT=-10;
temp=c0+c1*DT+c2*DT^2;
semilogy(nu_1,temp,'b-');
hold on;

DT=10;
temp=c0+c1*DT+c2*DT^2;
semilogy(nu_1,temp,'r-');
%%

nu_2=[27370:5:40800]';
coeff_2=repmat(nu_2*0,[1,3]);
files={'raw_BO3HH0.txt','raw_BO3HH1.txt','raw_BO3HH2.txt'};
for i=1:length(files)
    fileID=fopen(files{i}); % CKD coeffs
    temp=readContmData(fileID);
    if (i<2)
        coeff_2(:,i)=temp./nu_2;% c0
    else
        coeff_2(:,i)=temp; % c1/c0,c2/c0
    end
end
csvwrite('Contm_O3_b2.csv',[nu_2(1:end-1),coeff_2(1:end-1,:)]);

nu_3=[40800:100:54000]';
fileID=fopen('raw_BO3HUV.txt'); % correction factor for 2000-3000 cm-1, dnu=2 cm-1
temp=readContmData(fileID);
coeff_3=temp./nu_3;
plot(nu_3,coeff_3);
csvwrite('Contm_O3_b3.csv',[nu_3,coeff_3]);
%% process data for shortwave continuum -- O2
% Last modified: Mengying Li, 07/12/2018
close all
clear
clc

nu_1=[7536:2:8500]';% data blockbo2inf1
file='raw_BO2INF1.txt';
fileID=fopen(file); % CKD coeffs
temp=readContmData(fileID);
coeff_1=temp./nu_1; % remove radiation field
csvwrite('Contm_O2_b1.csv',[nu_1,coeff_1]);

nu_2=[9100:5:11000]'; % subroutine o2inf2
coeff_2=o2inf2(nu_2);
csvwrite('Contm_O2_b2.csv',[nu_2,coeff_2]);

nu_3=[12990.5:13229.5]'; % subroutine o2inf3
file='raw_BO2INF3.txt';
fileID=fopen(file); % CKD coeffs
temp=readContmData(fileID);
coeff_3=temp./nu_3; % remove radiation field
csvwrite('Contm_O2_b3.csv',[nu_3,coeff_3]);

nu_4=[15000:10:29870]'; % subroutine o2_vis
file='raw_BO2INVIS.txt';
fileID=fopen(file); % CKD coeffs
temp=readContmData(fileID);
xlosmt=2.68675*10^(19);
factor=xlosmt*10^(-20)*(55*273/296)^2;
factor=factor*89.5;
factor=1/factor;
coeff_4=temp*factor./nu_4; % remove radiation field
csvwrite('Contm_O2_b4.csv',[nu_4,coeff_4]);

nu_5=[36000:5:40000]'; % subroutine o2herz, Herzberg continuum
coeff_5=hertda(nu_5)./nu_5;
csvwrite('Contm_O2_b5.csv',[nu_5,coeff_5]);
