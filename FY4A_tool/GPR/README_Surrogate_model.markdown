This dataset comes from the shortwave radiative transfer model. 

Inputs and output of RTM model includes:

| Symbal | name                                                         | unit    | range   | Normalisation suggestion                                     |
| ------ | ------------------------------------------------------------ | ------- | ------- | ------------------------------------------------------------ |
| $T_a$  | surface Temp                                                 | K       | 294-306 |                                                              |
| Rh     | relative humidity                                            |         | 0-1     |                                                              |
| COD    | cloud optical depth                                          |         | 0-50    | suggest $e^{-\tau},\tau=COD$                                 |
| th0    | solar zenith angle                                           | Degree  | 0-60    | suggest $\cos(\theta_0)$ or use rad                          |
| dsw    | downwelling irradiation at surface or <br />GHI Global horizontal irradiation | W/m$^2$ | 50-1200 |                                                              |
| dni    | direct normal irradiance                                     | W/m$^2$ | 0-900   | have fixed the cos.                                          |
| dhi    | Diffuse horizontal irradiation                               | W/m$^2$ | 50-900  |                                                              |
| C01~06 | Normalised upwelling flux at top of atmosphere at each channel | 1       | 0-1     | Has normalized by the channel downwelling irraidance.<br />max of channel 1 is 1.0147. other are below 1. |
|        |                                                              |         |         |                                                              |

### Surrogate model 1 for cloud product:

Given $T_a, rh, COD，th0$

Target **C01,C02,C03,C04,C05,C06**.



### Surroate model 2 for ground solar product.

given $T_a, rh, th0,COD$

Target dsw, dni, dhi.





