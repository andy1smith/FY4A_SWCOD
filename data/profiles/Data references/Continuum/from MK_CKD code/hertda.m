function c=hertda(nu)
% O2 continuum in spectral range [36000,40000] cm-1
% rewrite from MT_CKD subrountine HERTDA
% Input a vector of wavenumbers
% Output a vector of coefficients
v1s=36000;
v2s=40000;
corr =((40000.-nu)/4000.)*7.917*10^(-27);
yratio = nu/48811.0;                  
c = 6.884*10^(-4)*(yratio).*exp(-69.738*(log(yratio)).^2)-corr;

c(nu<v1s | nu>v2s)=0; % assign zero to out of range       
end