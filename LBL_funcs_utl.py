import numpy as np
import math
from LBL_funcs_fullSpectrum import *
from LBL_funcs_inclined import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import shutil

# import plotting packages
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec

from mpl_toolkits.mplot3d import Axes3D # 3d plot
from matplotlib import cm #color map
from matplotlib.ticker import FormatStrFormatter # set decimals in ticks


from LBL_funcs_plotting import *
__all__ = [
    "Coeff_write_txt",
    "ghi2d_show",
    "polar_contour",
    "plot_3D_AngDist",
    "cloud_position_choice",
    "mcarats_job_create",
    "single_txt_replace",
]
def Coeff_write_txt(coeff_gas, coeff_aer, coeff_cld, kap, nu, index_nu, F_dw_os):
    ka_gas_M, ks_gas_M, g_gas_M = coeff_gas
    ka_aer_M, ks_aer_M, g_aer_M = coeff_aer
    ka_cld_M, ks_cld_M, g_cld_M = coeff_cld
    srcfle = '/Users/dengnan/Downloads/mcarats/validat/conf_source'
    file_path = f'/Users/dengnan/Downloads/mcarats/validat/conf_nu={int(nu[index_nu])}'
    # Check if the file exists
    if not os.path.exists(file_path):
        try:
            shutil.copy(srcfle, file_path)
            #print(f"File copied successfully to {file_path}")
        except Exception as e:
            print(f"Failed to copy file: {e}")
    single_txt_replace(file_path,int(nu[index_nu]),F_dw_os[index_nu])

    # aerosol
    ke_aer = (ka_aer_M[:, index_nu]+ks_aer_M[:, index_nu])*1e2 # format for scientific notation
    txt_replace(file_path, 'Atm_ext1d(1:, 2)', ke_aer)

    sca_aer = ks_aer_M[:, index_nu]*1e2/ke_aer
    sca_aer[np.isnan(sca_aer)] = 0
    txt_replace(file_path, 'Atm_omg1d(1:, 2)', sca_aer)

    g_aer = g_aer_M[:, index_nu]
    txt_replace(file_path, 'Atm_apf1d(1:, 2)', g_aer)

    # Gas
    rho_gas = ks_gas_M[:, index_nu] / (ka_gas_M[:, index_nu] + ks_gas_M[:, index_nu])
    txt_replace(file_path, 'Atm_omg1d(1:, 1)', rho_gas)

    ka_gas = ka_gas_M[:, index_nu]*1e2 # m-1
    txt_replace(file_path, 'Atm_abs1d(1:,1)', ka_gas)

    # cld
    ks_cld = ks_cld_M[kap, index_nu]*1e2 # cm-1-> m-1
    ka_cld = ka_cld_M[kap, index_nu]*1e2 # cm-1-> m-1
    g_cld  = g_cld_M[kap, index_nu]
    ke_cld = ks_cld + ka_cld # m-1
    sca_cld = ks_cld/ke_cld
    sca_cld[np.isnan(sca_cld)] = 0

    N_hL = len(kap) #
    filename = f'/Users/dengnan/Downloads/mcarats/validat/nu={int(nu[index_nu])}.atm'
    with open(filename, 'w') as f:  # Open file in write mode to create/overwrite file
        # Write the headers and data for tmpa3d
        f.write('%mdla3d\n')
        # Write the data for tmpa3d
        f.write('# tmpa3d (perturbations of atmospheric temperature)\n')
        for line in range(N_hL + 1):
            arr = np.zeros([60, 60])
            np.savetxt(f, arr.reshape(arr.shape[0], -1), fmt='%.2f', delimiter=' ')

        # Write the data for absa3d
        f.write('# absg3d (perturbations of absorption coefficient) for ikd=1\n')
        for line in range(N_hL):
            gridV = np.zeros([60, 60])
            gridV[20:40, 20:40] = np.zeros([20, 20]) + ka_cld[line]
            arr = gridV
            np.savetxt(f, arr.reshape(arr.shape[0], -1), fmt='%.5f', delimiter=' ')
            # f.write('\n')

        f.write('# extp3d (extinction coefficient of particles) for ip3d=1\n')
        for line in range(N_hL):
            gridV = np.zeros([60, 60])
            gridV[20:40, 20:40] = np.zeros([20, 20]) + ke_cld[line]
            arr = gridV
            np.savetxt(f, arr.reshape(arr.shape[0], -1), fmt='%.2f', delimiter=' ')

        # omgp3d data block
        f.write('# omgp3d (single scattering albedo of particles) for ip3d=1\n')
        for line in range(N_hL):
            gridV = np.zeros([60, 60])
            gridV[20:40, 20:40] = np.zeros([20, 20]) + sca_cld[line]
            arr = gridV
            np.savetxt(f, arr.reshape(arr.shape[0], -1), fmt='%.2f', delimiter=' ')

        # apfp3d data block
        f.write('# apfp3d (phase function specification parameter of particles) for ip3d=1\n')
        for line in range(N_hL):
            gridV = np.zeros([60, 60])
            gridV[20:40, 20:40] = np.zeros([20, 20]) + g_cld[line]
            arr = gridV
            np.savetxt(f, arr.reshape(arr.shape[0], -1), fmt='%.2f', delimiter=' ')
        return None

def txt_replace(file_path, sign_varaible, variable):
    formatted_variable = ', '.join(f"{x:.8e}" for x in variable)
    with open(file_path, 'r') as file:
        lines = file.readlines()
    # Prepare to modify lines
    start_line = None
    for index, line in enumerate(lines):
        if sign_varaible in line:
            start_line = index

    # Replace the specified section with new data
    if start_line is not None:
        lines[start_line] = sign_varaible + '=' + f"{formatted_variable},\n"
    else:
        print(sign_varaible)
        print('Fail to write')

    # Write back to the file
    with open(file_path, 'w') as file:
        file.writelines(lines)
    return None

def single_txt_replace(file_path, nu_, F_dw_os_):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    # Prepare to modify lines
    atm_line = None
    src_line = None
    for index, line in enumerate(lines):
        if 'Atm_inpfile' in line:
            atm_line = index
        if 'rc_flx' in line:
            src_line = index

    # Replace the specified section with new data
    if atm_line is not None:
        lines[atm_line] = 'Atm_inpfile' + '=' + f"\'nu={int(nu_)}.atm\'\n"
    else:
        print('Fail to write')

    if src_line is not None:
        formatted_variable = str(F_dw_os_) + ', ' + str(F_dw_os_)
        lines[src_line] = 'Src_flx' + '=' + f"{formatted_variable}\n"
    else:
        print('Fail to write')

    # Write back to the file
    with open(file_path, 'w') as file:
        file.writelines(lines)
    return None

def mcarats_job_create(nu,job_name):
    headers = [
        "set pfx=out",
        "set ptot_ci=1e4 # total number of incident photons",
        "set ptot_sr=1e5",
        "set ptot_tr=1e6",
        "set exec=mcarats"
    ]
    # Open a text file to write
    with open(job_name, "w") as file:
        # Write header lines
        for line in headers:
            file.write(line + '\n')

        file.write('\n')  # Add a newline to separate headers from other content for clarity

        # Write other contents with variable substitution from the list nu
        for index in range(len(nu)):
            content_line = f"$exec $ptot_sr 0 conf_nu={int(nu[index])} $pfx\"_nu={int(nu[index])}\""
            file.write(content_line + '\n')
    return None

def cloud_position_choice(N_layer,model,kap,th0_v,phi0,Mute=True):
    # N_layer = 54
    # model = 'AFGL midlatitude summer'
    # kap = [[8, 9, 10, 11, 12, 13]]
    theta0_v = th0_v / 180 * math.pi  # solar zenith angle in rad

    p, pa = set_pressure(N_layer)
    z, za = set_height(model, p, pa)
    dz_cld = z[kap[-1] + 1] - z[kap[0]]  # in m
    if Mute == True:
        print(f"cloud bottom = {z[kap[0]]} m")
        print(f"cloud top = {z[kap[-1] + 1]} m")
        print(f"cloud depth = {dz_cld} m")
        print('---------------------------------------')
        print(f"if the TOA = 120 km, theta = {th0_v},\nthen corrdinates of cloud center:")
    cld_top = (z[kap[-1] + 1]) / 1e3
    cld_botm = (z[kap[0]]) / 1e3
    if Mute == True:
        print(f"\t- for top = {cld_top:.2f} km\n\t- for bottom = {cld_botm:2f} km")
        print('----------------------------------------------------')
        print(f"projection of the cloud:")
    top_prj = cld_top * np.tan(theta0_v) * np.cos(phi0)
    btom_prj = cld_botm * np.tan(theta0_v) * np.cos(phi0)
    x_cld = 0.5*(abs(top_prj)-abs(btom_prj))
    if Mute == True:
        print(f"\tthe suggest xcld,ycld = {-x_cld},0")
    return cld_top


def polar_contour(theta0, Mrxyz, nu, F_dw_os, N_bundles, testmode, is_flux=True, downwelling=True):
    # *****switch between flux and intensity
    font = 15
    fontfml = 'Times New Roman'
    plt.rcParams['font.size'] = font
    plt.rcParams['font.family'] = fontfml
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = fontfml
    plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
    plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
    # costimize cmap
    # colors = [(255,255,255), (198,219,239),(158,202,225),(107,174,214), # white -> blue
    #          (65,182,196),(127,205,187),(199,233,180), (237,248,177),(255,255,204),# blue->green->yellow
    #           (255,255,178),(254,217,118),(254,178,76),(253,141,60),(240,59,32),(189,0,38)] # yellow->orange->red
    # colors = [(255,255,255), (198,219,239),(158,202,225),(107,174,214),
    #           (0, 87, 255),
    #           (0, 140, 255),(0, 194, 255),(255, 255, 3),(255, 201, 0),
    #           (255, 146, 0),(255, 90, 0),(255, 34, 0),(233, 0, 0),
    #           (177, 0, 0),(122, 0, 0),(66, 0, 0),(10, 0, 0),(0,0,0)]
    #my_cmap = make_cmap(colors, bit=True)
    #theta0 = theta0 / 180 * math.pi
    phi0 = 0 / 180 * math.pi
    # titles=['only behind cloud','behind cloud+no cloud',"no clouds"]
    # titles=['(a) dcldx = -5.0','(b) dcldx = 0','(c) dcldx = 10']

    fig = plt.figure(figsize=(5, 5))#, dpi=300)
    fontfml = 'Times New Roman'
    gs1 = gridspec.GridSpec(1, 1)
    gs1.update(wspace=0.6, hspace=0)  # set the spacing between axes.
    d_th = 2
    d_phi = 5
    bins_theta = np.arange(0, 91, d_th)
    bins_phi = np.arange(-180, 181, d_phi)  # to include the binedges=180 when plotting
    vmin = 0.5
    vmax = 3.
    levels = np.linspace(vmin, vmax, 25)
    # for ifile in range(3):
    for ifile in range(0, 1):
        # uw_rxyz_M=np.load("results_shortwave/inclined/uw_rxyz_AOD=0.05"+files[ifile]) #******
        #dw_rxyz_M = np.load(file_dir + Fls[ifile], allow_pickle=True)
        #nu = dw_rxyz_M.item().get('gk')
        dw_rx, dw_ry, dw_rz, uw_rx, uw_ry, uw_rz = [np.zeros((2*N_bundles + 10, len(nu))) * np.nan for i in range(0, 6)]
        #Mrxyz = dw_rxyz_M.item().get('dw_rxyz_M')
        #F_dw_os = dw_rxyz_M.item().get('F_dw_os')
        H = np.zeros((len(bins_phi) - 1, len(bins_theta) - 1))
        for k in range(len(F_dw_os)):
            dw_rxyz = Mrxyz[k]
            N_dw = len(dw_rxyz)
            dw_rx[0:N_dw, k] = np.array([x[0] for x in dw_rxyz])
            dw_ry[0:N_dw, k] = np.array([x[1] for x in dw_rxyz])
            dw_rz[0:N_dw, k] = np.array([x[2] for x in dw_rxyz])
            if downwelling :
                theta_v, phi_v = theta_phi(dw_rx[:, k], dw_ry[:, k], -dw_rz[:, k])
            else:
                theta_v, phi_v = theta_phi(dw_rx[:, k], dw_ry[:, k], dw_rz[:, k])
            ind = np.isnan(phi_v)
            theta_v = theta_v[~ind]
            phi_v = phi_v[~ind] - phi0
            phi_v[phi_v > math.pi] -= 2 * math.pi
            # newly added. ======== comment it if not need =========!!!!!!!!!!!!!!!!!!!
            # phi_v,theta_v=extract_cloud_points(phi_v,theta_v)
            # Compute the 2D histogram [rad, deg]
            H_k, Pedges, Tedges = np.histogram2d(phi_v, np.rad2deg(theta_v),
                                                 bins=(np.radians(bins_phi), bins_theta))  # newly added.
            if testmode == 'mono':
                H += H_k * F_dw_os * np.cos(theta0) / N_bundles
            else:
                H += H_k * F_dw_os[k] * np.cos(theta0) / N_bundles  # 3 is dnu
        print('Polar GHI = ', np.sum(H), 'W/m2')
        for ibeta in range(0, 1):
            phi, theta = np.meshgrid(Pedges, Tedges[:-1])  # [rad, deg]
            if (not is_flux):
                ths = np.deg2rad(theta.T + d_th / 2)  # rad
                H /= 0.5 * np.sin(2 * ths[:-1])  # **** change between flux and intensity, comment for flux
            H /= np.deg2rad(d_th) * np.deg2rad(d_phi)  # per solid angle, in the direction of beam
            Z = np.log10(H.T + 1.0)  # remove zero
            # print("mean F",np.mean(H))
            # print("max F",np.amax(H))
            # print("sum_F", np.sum(H))
            print('---------')
            #print(H[14,36]/np.sum(H))

            # Create the plot
            ax1 = fig.add_subplot(gs1[ibeta, ifile], projection='polar')
            # Plot the data
            Z = np.concatenate((Z, Z[:, -1:]), axis=1)
            contour = ax1.contourf(phi, theta, Z, levels=levels, cmap='jet',extend='both')  # [rad, deg]
            ax1.set_xticks(np.pi / 180. * np.linspace(0, 360, 8, endpoint=False))  # [rad, -]
            ax1.set_xticklabels(['0°', '45°', '90°', '135°', '180°', '-135°', '-90°', '-145°'],
                                fontsize=font, family=fontfml)
            # Set the radial ticks and labels
            ax1.set_yticks([0, 30, 60, 90])
            ax1.set_yticklabels(['0°', '30°', '60°', '90°'])#,fontsize=font, family=fontfml)
            #ax1.set_title(titles[ifile], x=0.5, y=1.15, fontsize=font, family=fontfml)
            ax1.tick_params(axis='y', labelcolor='White')
            ax1.grid(True, linestyle=':', color='White', alpha=0.85)  # Set grid lines to dashed
            ax1.set_rlabel_position(90 + 25)
            #cbar_ax1 = fig.add_axes([0.17, 0.01, 0.7, 0.02])
            # cbar_ax1 = fig.add_axes([0.1 + 0.3 * ifile, 0.1, 0.02, 0.7])
            # cbar1 = plt.colorbar(c, cax=cbar_ax1, orientation='vertical')
            colorbar_ticks = [0.5, 1, 1.5, 2, 2.5, 3]  # 5 ticks from min to max
            cbar1 = plt.colorbar(contour, ax=ax1, ticks=colorbar_ticks, orientation='horizontal')
            #cbar1 = plt.colorbar(contour, ax=ax1, orientation='horizontal')
            cbar1.set_ticks(colorbar_ticks)#, fontsize=font - 2, family=fontfml)
            cbar1.set_ticklabels([f'{tick:.1f}' for tick in colorbar_ticks])
        if ifile == 0:
            Z1 = Z
        elif ifile == 1:
            Z2 = Z
    if (is_flux):
        cbar1.set_label('log$_{10}$ of flux [W m$^{-2}$]', fontsize=font, family=fontfml, rotation=0,
                        labelpad=0)  # *****
        figname = f"cloudcover_COD=10.png"
        fig.savefig(figname, dpi=300, bbox_inches='tight',transparent=True)  #***
    else:
        cbar1.set_label('log$_{10}$ of intensity [W m$^{-2}$ sr$^{-1}$]', fontsize=font, family=fontfml, rotation=0,
                        labelpad=0)  # ****
        #fig.savefig(figname, dpi=300, bbox_inches='tight')  # ***
    # plot_diff(phi, theta, Z1-Z2)
    plt.show()
    return None

def ghi2d_show(F_ghi_2d,xmax,cld_r):

    fig = plt.figure(figsize=(4, 4))
    gs1 = gridspec.GridSpec(1, 1)
    gs1.update(wspace=0.1, hspace=0.3)
    Z = F_ghi_2d#np.log10(F_ghi_2d.T + 1.0)
    ax1 = fig.add_subplot(gs1[0,0])
    im = ax1.imshow(Z, cmap='Spectral_r', origin="lower")#, #norm=colors.LogNorm())
    ax1.set_xlabel('x [km]')
    ax1.set_ylabel('y [bin number]')

    ax1.set_xticks(np.linspace(0,F_ghi_2d[0].shape[0],6))
    #ax1.set_xticklabels(np.linspace(-xmax, xmax, 6))
    cbar1 = plt.colorbar(im)
    ax1.set_title(f'sun_r = cld_r = {cld_r}')
    #colorbar_ticks = [0.001, 0.01, 0.1, 1]  # 5 ticks from min to max
    #cbar1.set_ticks(colorbar_ticks)
    #fig.savefig('cldside_COD=10_extract.png', dpi=300, bbox_inches='tight', transparent=True)
    plt.show()
    return None


def plot_3D_AngDist(vmax, COD, theta0, phi0, Mrxyz, nu, F_dw_os, N_bundles, testmode, is_flux=False,
                    Norm=False):  # Z_csky
    ####################### formatting the plot#################################
    font = 15
    fontfml = 'Times New Roman'
    plt.rcParams['font.size'] = font
    plt.rcParams['font.family'] = fontfml
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = fontfml
    plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
    plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
    # costimize cmap
    colors = [(255, 255, 255), (198, 219, 239), (158, 202, 225), (107, 174, 214),  # white -> blue
              (65, 182, 196), (127, 205, 187), (199, 233, 180), (237, 248, 177), (255, 255, 204),  # blue->green->yellow
              (255, 255, 178), (254, 217, 118), (254, 178, 76), (253, 141, 60), (240, 59, 32),
              (189, 0, 38)]  # yellow->orange->red

    my_cmap = make_cmap(colors, bit=True)
    theta0 = theta0 / 180 * math.pi
    phi0 = 0 / 180 * math.pi

    # Create a figure of size y*y inches, 600 dots per inch
    fig = plt.figure(figsize=(6, 5))  # , dpi=300)
    font = 13
    fontfml = 'Times New Roman'
    gs1 = gridspec.GridSpec(1, 1)
    gs1.update(wspace=-0.1, hspace=0)  # set the spacing between axes.
    d_th = 2
    d_phi = 5
    bins_theta = np.arange(0, 91, d_th)
    bins_phi = np.arange(-180, 181, d_phi)
    for ifile in range(1):
        # results=np.load(file_dir+Fls[1],allow_pickle=True).item() #******
        dw_rxyz_M = Mrxyz  # results.get('dw_rxyz')
        dw_rx, dw_ry, dw_rz, uw_rx, uw_ry, uw_rz = [np.zeros((N_bundles + 10, len(nu))) * np.nan for i in range(0, 6)]
        H = np.zeros((len(bins_theta) - 1, len(bins_phi) - 1))
        for k in range(0, len(nu)):
            dw_rxyz = dw_rxyz_M[k]
            N_dw = len(dw_rxyz)
            dw_rx[0:N_dw, k] = np.array([x[0] for x in dw_rxyz])
            dw_ry[0:N_dw, k] = np.array([x[1] for x in dw_rxyz])
            dw_rz[0:N_dw, k] = np.array([x[2] for x in dw_rxyz])
            theta_v, phi_v = theta_phi(dw_rx[:, k], dw_ry[:, k], dw_rz[:, k])
            ind = np.isnan(phi_v)
            theta_v = theta_v[~ind]
            phi_v = phi_v[~ind] - phi0
            phi_v[phi_v > math.pi] -= 2 * math.pi
            H_k, xedges, yedges = np.histogram2d(np.rad2deg(theta_v), np.rad2deg(phi_v), bins=(bins_theta, bins_phi))
            if Norm == True:
                H += H_k * np.cos(theta0) * 1 / N_bundles  # *F_dw_os[k]*3/N_bundles # 3 is dnu
            else:
                H += H_k * np.cos(theta0) * F_dw_os[k] * 3 / N_bundles
        for ibeta in range(0, 1):
            theta_, phi_ = np.meshgrid(xedges[0:-1], yedges[0:-1])
            if (not is_flux):
                ths = np.deg2rad(theta_.T + d_th / 2)  # rad dw # division 2 for the 2sintcost
                H /= np.sin(ths)  # 0.5 * np.sin(2 * ths)
            H /= np.deg2rad(d_th) * np.deg2rad(d_phi)  # per solid angle, in the direction of beam
            if Norm == True:
                Z = H.T / N_bundles
            else:
                Z = np.log10(H.T + 1.0)  # remove zero
            # print('min',np.min(Z))
            # print('max',np.max(Z))
            print('min', np.min(H))
            print('max', np.max(H))
            # print(H[14, 36] / np.sum(H))

            # plot 3D distribution ---- split from here
            # to enable colorbar
            ax2 = fig.add_subplot(gs1[ibeta, ifile], projection='3d')
            kwargs = dict(cmap=my_cmap, alpha=1.0, vmin=0, vmax=vmax, linewidth=0, antialiased=False)
            p1 = ax2.plot_surface(phi_, theta_, Z, **kwargs)
            ax2.axis('off')
            ax2.set_visible(False)

            ax1 = fig.add_subplot(gs1[ibeta, ifile], projection='3d')
            cmap = my_cmap
            max_height = vmax  # get range of colorbars so we can normalize
            min_height = 0
            rgba = [cmap((k - min_height) / max_height) for k in
                    Z.ravel()]  # scale each z to [0,1], and get their rgb values
            kwargs = dict(color=rgba, zsort='average', edgecolor='none', shade=False)
            p2 = ax1.bar3d(phi_.ravel(), theta_.ravel(), Z.ravel() * 0., d_th, d_phi, Z.ravel(), **kwargs)
            ax1.view_init(15, -20)  # view angle of 3D surface

            format_axes(ax1, [-180, 180], [0, 90], [-180, -90, 0, 90, 180], [0, 15, 30, 45, 60, 75, 90], False, False)
            ax1.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # set panes white color
            ax1.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            ax1.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

            ax1.xaxis.set_ticklabels(['', -90, 0, 90, ''], fontsize=font - 2, family=fontfml, rotation=30)
            ax1.tick_params(axis='x', which='major', pad=-7)
            ax1.set_xlabel('$\\gamma-\\gamma_{az}$ [$^{\circ}$]', fontsize=font, family=fontfml, labelpad=-5)

            ax1.yaxis.set_ticklabels([0, 15, 30, 45, 60, 75, 90], fontsize=font - 2, family=fontfml)
            ax1.tick_params(axis='y', which='major', pad=-7)
            ax1.set_ylabel('$\\theta$ [$^{\circ}$]', fontsize=font, family=fontfml, labelpad=-5)
            if Norm == True:
                ax1.set_zlim(0, 1)
                ax1.zaxis.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
                ax1.zaxis.set_ticklabels(['', 0.2, 0.4, 0.6, 0.8, 1], fontsize=font - 2, family=fontfml)
            else:
                ax1.set_zlim(0, 3)
                ax1.zaxis.set_ticks([0, 0.5, 1, 1.5, 2, 2.5, 3])
                ax1.zaxis.set_ticklabels(['', 0.5, 1, 1.5, 2, 2.5, 3], fontsize=font - 2, family=fontfml)
            ax1.tick_params(axis='z', which='major', pad=-3)

            if (ifile == 2):
                if (is_flux):
                    ax1.set_zlabel('log$_{10}$ of flux [W m$^{-2}$]', fontsize=font, family=fontfml,
                                   labelpad=-10)  # *****
                else:
                    ax1.set_zlabel('log$_{10}$ of intensity [W m$^{-2}$ sr$^{-1}$]', fontsize=font,
                                   family=fontfml, labelpad=-10)  # ****
        ax1.set_title(f"COD = {COD}", x=0.5, y=0.9, fontsize=font, family=fontfml)
        # ax1.set_title(f"Z = {np.round(np.rad2deg(theta0))}",x=0.5,y=0.9,fontsize=font,family=fontfml)
        cbar_ax1 = fig.add_axes([0.85, 0.2, 0.02, 0.5])  # [left, bottom, width, height]
        cbar1 = plt.colorbar(p1, cax=cbar_ax1, orientation='vertical')
        if (is_flux):
            cbar1.set_label('log$_{10}$ of flux [W m$^{-2}$]', rotation=0, labelpad=0)  # *****
            # fig.savefig('phi_rxyz_3D_Flux.eps', dpi=300, bbox_inches='tight')#***
        else:
            fig_dir = './figures/'
            if Norm == False:
                cbar1.set_label('log$_{10}$ of intensity [W m$^{-2}$ sr$^{-1}$]', rotation=90,
                                labelpad=0, fontsize=font, family=fontfml)  # ****
                figname = fig_dir + f"I_angular_COD={COD}_Z={np.round(np.rad2deg(theta0))}.png"
            else:
                figname = fig_dir + f"NorI_angular_COD={COD}_Z={np.round(np.rad2deg(theta0))}.png"
            #fig.savefig(figname, dpi=300, bbox_inches='tight')  # ***
    plt.show()
    return H