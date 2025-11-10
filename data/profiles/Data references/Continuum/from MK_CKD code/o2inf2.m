function c=o2inf2(nu)
% O2 continuum in spectral range [9100,11000] cm-1
% rewrite from MT_CKD subrountine o2inf2
% Input a vector of wavenumbers
% Output a vector of coefficients
v1_osc=9375; hw1=58.96; v2_osc=9439; hw2=45.04;
s1=1.166*10^(-4); s2=3.086*10^(-5);
v1s=9100;
v2s=11000;

dv1=nu-v1_osc;
dv2=nu-v2_osc;
damp1=1.0+nu*0;
damp1(dv1<0)=exp(dv1(dv1<0)/176.1);
damp2=1.0+nu*0;
damp2(dv2<0)=exp(dv2(dv2<0)/176.1);
o2inf = 0.31831*((s1*damp1/hw1)./(1.+(dv1/hw1).^2)+...
            (s2*damp2/hw2)./(1.+(dv2/hw2).^2))*1.054;

c = o2inf./nu; % remove radiation field

c(nu<v1s | nu>v2s)=0; % assign zero to out of range
end