from main_SW_scope_nearealtime import *
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression


def load_data(data_dir, sat, sky):
    Sat_dir = Sat_preprocess(data_dir, sat=sat)
    sat = pd.read_hdf(os.path.join(data_dir + Sat_dir, 
                       "GOES_day_BON_radiance_satellite_June_COD>10_water_{}.h5".format(sky)),
                      'df')
    return sat


def preprocess_data(sat):
    channels = ['C{:02d}'.format(c) for c in range(1, 6 + 1)]
    try:
        sat_ref = sat[channels]
    except KeyError:
        sat_ref = sat['Radiance']

    Sat_nor_df = pd.DataFrame()
    Rc_rtm_df = pd.DataFrame()
    RTM_nor_df = pd.DataFrame()

    for i in range(sat_ref.shape[0]):
        Sun_Zen = sat['Sun_Zen'][i]
        T_a = sat['temp'][i] + 273.15
        RH = sat['rh'][i]
        local_zen = sat.get('Local_Ze', sat.get('local_Zen'))[i]
        rela_azi = sat['rela_azi'][i]
        COD = int(sat['COD'][i])

        Sat_nor_df = pd.concat([Sat_nor_df, min_max_nor(sat_ref.iloc[i].to_frame().T)])

        Rc_rtm = nearealtime_RTM(Sun_Zen, local_zen, rela_azi, COD, T_a, RH, channels, file_dir=data_dir,
                                 bandmode='GOES', N_bundles=10000)
        RTM_nor_df = pd.concat([RTM_nor_df, min_max_nor(Rc_rtm)])
        Rc_rtm_df = pd.concat([Rc_rtm_df, Rc_rtm], ignore_index=True)

    return sat_ref, Sat_nor_df, Rc_rtm_df, RTM_nor_df


def plot_data(sat_ref, Rc_rtm_df, channels, figname):
    font = 13
    fontfml = 'Times New Roman'
    plt.rcParams['font.size'] = font
    plt.rcParams['font.family'] = fontfml
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = fontfml
    plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
    plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

    fig = plt.figure(figsize=(12, 6))
    gs1 = gridspec.GridSpec(2, 3)
    gs1.update(wspace=0.15, hspace=0.1)

    for idx, ch in enumerate(channels):
        ax = fig.add_subplot(gs1[idx // 3, idx % 3])
        try :
            x = sat_ref[ch].values
        except KeyError :
            x = sat_ref.loc[ch].values
        y = Rc_rtm_df[ch].values
        model = LinearRegression().fit(x.reshape(-1, 1), y)
        y_pred = model.predict(x.reshape(-1, 1))
        r2 = r2_score(x, y)
        mae = np.mean(np.abs(x - y))
        rmse = np.sqrt(np.mean((x - y) ** 2))
        bias = np.mean(y - x)
        slope = model.coef_[0]
        intercept = model.intercept_
        min_val = min(x.min(), y.min())
        max_val = max(x.max(), y.max())

        sns.scatterplot(x=x, y=y, ax=ax, color='k', alpha=0.8, s=18)
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        ax.plot(x, y_pred, color='blue', linestyle='-', linewidth=1.5, label='Regression')
        ax.set_xlim(min_val * 0.99, max_val * 1.01)
        ax.set_ylim(min_val * 0.99, max_val * 1.01)

        if ch == 'C04':
            ax.set_xlim(min_val - 0.01, max_val * 1.01)
            ax.set_ylim(min_val - 0.01, max_val * 1.01)

        stats_text = (
            f'{ch}\n'
            f'R² = {r2:.3f}\n'
            # f'MAE = {mae:.3f}\n'
            f'RMSE = {rmse:.3f}\n'
            # f'Bias = {bias:.3f}\n'
            f'Slope = {slope:.3f}\n'
            f'Intercept = {intercept:.3f}'
        )

        if ch == 'C04':
            ax.text(0.5, 0.45, stats_text, transform=ax.transAxes, fontsize=13, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        else:
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=13, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        if idx // 3 == 1:
            ax.set_xlabel(r'Satellite [W/(m$^2$ sr)]', fontsize=font, family=fontfml)
        else:
            ax.set_xticklabels([])

        if ch == 'C01' or ch == 'C04':
            ax.set_ylabel('RTM [W/(m$^2$ sr)]', fontsize=font, family=fontfml)
        ax.grid(color='grey', linestyle='--', linewidth=0.5)
        # ax.legend(loc='lower right', fontsize=10)

    fig_dir = './figures/'

    fig.savefig(fig_dir + figname, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    data_dir = './GOES_data/'
    sky = "day"
    year,month = '2019', 'June'
    sat = 'GOES16'
    phase = 'water'
    save_path = './GOES_validation/{year}_{month}_water/'.format(year=year, month=month)
    channels = ['C{:02d}'.format(c) for c in range(1, 6 + 1)]

    Run = True
    if Run:
        sat = load_data(data_dir, sat, sky)
        sat_ref, Sat_nor_df, Rc_rtm_df, RTM_nor_df = preprocess_data(sat, sys)

        if not os.path.isdir(save_path):
            print("Creating directory:", save_path)
            os.mkdir(save_path)
        sat_ref.to_hdf(os.path.join(save_path, f'sat_ref_{year}_{month}.h5'), key='df', mode='w')
        Sat_nor_df.to_hdf(os.path.join(save_path, f'Sat_nor_df_{year}_{month}.h5'), key='df', mode='w')
        Rc_rtm_df.to_hdf(os.path.join(save_path, f'Rc_rtm_df_{year}_{month}.h5'), key='df', mode='w')
        RTM_nor_df.to_hdf(os.path.join(save_path, f'RTM_nor_df_{year}_{month}.h5'), key='df', mode='w')
    else:
        # Load the saved data
        sat_ref = pd.read_hdf(os.path.join(save_path, f'sat_ref_{year}_{month}.h5'), key='df')
        Sat_nor_df = pd.read_hdf(os.path.join(save_path, f'Sat_nor_df_{year}_{month}.h5'), key='df')
        Rc_rtm_df = pd.read_hdf(os.path.join(save_path, f'Rc_rtm_df_{year}_{month}.h5'), key='df')
        RTM_nor_df = pd.read_hdf(os.path.join(save_path, f'RTM_nor_df_{year}_{month}.h5'), key='df')
    print('Finish!')
    #figname = 'Bias_Nor_BON_June_GOES_RTM_17.png'
    #plot_data(Sat_nor_df, RTM_nor_df, channels, figname)
    #plot_data(sat_ref, Rc_rtm_df, channels)
