import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt


class AngularDistributionModel:
    def __init__(self):
        self.cod_bins = [0.5, 1, 2, 5, 10, 20, 30, 50]
        self.sza_bins = [0, 15, 30, 45, 60, 75]
        self.channels = ['C01', 'C02', 'C03', 'C04', 'C05', 'C06']

    def angular_kernels(self, theta, phi):
        """Define angular kernels for decomposition"""
        theta_rad = np.radians(theta)
        phi_rad = np.radians(phi)

        # Kernel 1: Lambertian (isotropic)
        k1 = np.ones_like(theta)

        # Kernel 2: Limb darkening
        k2 = np.cos(theta_rad)

        # Kernel 3: Forward scattering (simple approximation)
        # Peak near phi=0 (forward scatter)
        k3 = np.exp(-((phi_rad) ** 2) / (2 * (np.pi / 4) ** 2))

        # Kernel 4: Hotspot (backscatter enhancement)
        # Peak near phi=180
        phi_back = np.abs(phi_rad - np.pi)
        k4 = np.exp(-(phi_back ** 2) / (2 * (np.pi / 6) ** 2))

        return np.array([k1, k2, k3, k4])

    def fit_kernels_to_reflectance(self, theta_grid, phi_grid, reflectance_2d):
        """Fit kernel coefficients to 2D reflectance map"""
        # Flatten the grids
        theta_flat = theta_grid.flatten()
        phi_flat = phi_grid.flatten()
        refl_flat = reflectance_2d.flatten()

        # Get kernels for all points
        kernels = self.angular_kernels(theta_flat, phi_flat)

        # Fit coefficients using least squares
        def model_func(dummy, c0, c1, c2, c3):
            return c0 * kernels[0] + c1 * kernels[1] + c2 * kernels[2] + c3 * kernels[3]

        # Fit the model
        popt, pcov = curve_fit(model_func, np.zeros_like(refl_flat), refl_flat)

        return popt, np.sqrt(np.diag(pcov))

    def smooth_monte_carlo_data(self, reflectance_2d, kernel_size=3):
        """Apply smoothing to reduce Monte Carlo noise"""
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(reflectance_2d, sigma=kernel_size)

    def build_coefficient_lut(self, monte_carlo_data):
        """
        Build lookup table of kernel coefficients

        monte_carlo_data: dict with structure
        {
            'channel': {
                'cod': {
                    'sza': {
                        'theta_grid': 2D array,
                        'phi_grid': 2D array,
                        'reflectance': 2D array
                    }
                }
            }
        }
        """
        coefficient_lut = {}

        for channel in self.channels:
            coefficient_lut[channel] = {}

            for cod in self.cod_bins:
                coefficient_lut[channel][cod] = {}

                for sza in self.sza_bins:
                    if (channel in monte_carlo_data and
                            cod in monte_carlo_data[channel] and
                            sza in monte_carlo_data[channel][cod]):
                        data = monte_carlo_data[channel][cod][sza]

                        # Smooth the noisy Monte Carlo data
                        smoothed_refl = self.smooth_monte_carlo_data(data['reflectance'])

                        # Fit kernels
                        coeffs, errors = self.fit_kernels_to_reflectance(
                            data['theta_grid'], data['phi_grid'], smoothed_refl
                        )

                        coefficient_lut[channel][cod][sza] = {
                            'coefficients': coeffs,
                            'errors': errors
                        }

        return coefficient_lut

    def interpolate_coefficients(self, coefficient_lut, channel, cod_target, sza_target):
        """Interpolate coefficients for arbitrary COD and SZA"""

        # Extract coefficient arrays for interpolation
        cod_array = np.array(self.cod_bins)
        sza_array = np.array(self.sza_bins)

        # Get coefficient grids
        coeff_grids = []
        for i in range(4):  # 4 kernels
            coeff_grid = np.zeros((len(self.cod_bins), len(self.sza_bins)))
            for ci, cod in enumerate(self.cod_bins):
                for si, sza in enumerate(self.sza_bins):
                    if cod in coefficient_lut[channel] and sza in coefficient_lut[channel][cod]:
                        coeff_grid[ci, si] = coefficient_lut[channel][cod][sza]['coefficients'][i]
            coeff_grids.append(coeff_grid)

        # Create interpolators
        interpolated_coeffs = []
        for coeff_grid in coeff_grids:
            interp = RegularGridInterpolator(
                (cod_array, sza_array), coeff_grid,
                bounds_error=False, fill_value=0
            )
            interpolated_coeffs.append(interp([cod_target, sza_target])[0])

        return np.array(interpolated_coeffs)

    def compute_reflectance(self, theta, phi, coefficients):
        """Compute reflectance for given angles using fitted coefficients"""
        kernels = self.angular_kernels(theta, phi)
        return np.sum(coefficients[:, np.newaxis, np.newaxis] * kernels, axis=0)

    def validate_model(self, coefficient_lut, validation_data):
        """Validate the ADM against held-out data"""
        rmse_results = {}

        for channel in validation_data:
            rmse_results[channel] = {}
            for cod in validation_data[channel]:
                rmse_results[channel][cod] = {}
                for sza in validation_data[channel][cod]:
                    # Get interpolated coefficients
                    coeffs = self.interpolate_coefficients(
                        coefficient_lut, channel, cod, sza
                    )

                    # Compute model prediction
                    data = validation_data[channel][cod][sza]
                    pred_refl = self.compute_reflectance(
                        data['theta_grid'], data['phi_grid'], coeffs
                    )

                    # Calculate RMSE
                    rmse = np.sqrt(np.mean((pred_refl - data['reflectance']) ** 2))
                    rmse_results[channel][cod][sza] = rmse

        return rmse_results


# Example usage
def example_usage():
    """Example of how to use the ADM builder"""

    # Initialize the model
    adm = AngularDistributionModel()

    # Example: Create synthetic data (replace with your Monte Carlo results)
    monte_carlo_data = {}

    # You would populate this with your actual Monte Carlo simulation results
    # monte_carlo_data['C01'][10][30] = {
    #     'theta_grid': theta_2d_array,
    #     'phi_grid': phi_2d_array,
    #     'reflectance': reflectance_2d_array
    # }

    # Build the coefficient LUT
    # coefficient_lut = adm.build_coefficient_lut(monte_carlo_data)

    # Example: Interpolate for arbitrary COD and SZA
    # coeffs = adm.interpolate_coefficients(coefficient_lut, 'C01', 15.5, 37.2)

    # Compute reflectance for new viewing geometry
    # theta_new, phi_new = np.meshgrid(np.arange(0, 90, 5), np.arange(-180, 180, 10))
    # reflectance_new = adm.compute_reflectance(theta_new, phi_new, coeffs)

    print("ADM framework ready for implementation")


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


class AnalyticalADF:
    """Analytical Angular Distribution Functions based on physical scattering theory"""

    def __init__(self):
        # Asymmetry parameter for water droplets (typically 0.85-0.88)
        self.g_water = 0.85

    def henyey_greenberg_phase(self, phi, g):
        """Henyey-Greenberg phase function for azimuth dependence"""
        phi_rad = np.radians(phi)
        cos_phi = np.cos(phi_rad)
        return (1 - g ** 2) / ((1 + g ** 2 - 2 * g * cos_phi) ** (3 / 2))

    def channel_specific_zenith_functions(self, theta, channel, params):
        """Channel-specific zenith angle dependence functions"""
        theta_rad = np.radians(theta)

        if channel == 'C01':
            # Limb brightening: sec(θ) with enhancement factor
            return params['a'] / np.cos(theta_rad) + params['b']

        elif channel in ['C02', 'C05']:
            # Bell-shaped: sin(2θ) pattern
            return params['a'] * np.sin(2 * theta_rad) + params['b']

        elif channel in ['C03', 'C06']:
            # Broad peak: modified Gaussian with sec(θ) factor
            gaussian = np.exp(-(theta_rad - params['theta0']) ** 2 / (2 * params['sigma'] ** 2))
            sec_factor = 1 / np.cos(theta_rad)
            return params['a'] * gaussian * sec_factor + params['b']

        elif channel == 'C04':
            # Nearly constant (Lambertian)
            return params['a'] * np.ones_like(theta) + params['b']

        else:
            raise ValueError(f"Unknown channel: {channel}")

    def combined_angular_function(self, theta, phi, channel, zenith_params, azimuth_params):
        """Combined angular distribution function R(θ,φ)"""

        # Zenith component
        I_theta = self.channel_specific_zenith_functions(theta, channel, zenith_params)

        # Azimuth component (Henyey-Greenberg)
        I_phi = self.henyey_greenberg_phase(phi, azimuth_params['g'])

        # Additional azimuth modulation
        phi_rad = np.radians(phi)

        # Forward scattering enhancement
        forward_peak = azimuth_params['f_amp'] * np.exp(-phi_rad ** 2 / (2 * azimuth_params['f_width'] ** 2))

        # Backscatter enhancement
        back_phi = np.abs(phi_rad - np.pi)
        back_peak = azimuth_params['b_amp'] * np.exp(-back_phi ** 2 / (2 * azimuth_params['b_width'] ** 2))

        # Combine components
        I_phi_total = I_phi + forward_peak + back_peak

        # Normalization factor (depends on viewing geometry)
        norm_factor = azimuth_params['norm']

        return I_theta * I_phi_total * norm_factor

    def fit_zenith_function(self, theta_data, intensity_data, channel):
        """Fit zenith angle dependence to data"""

        if channel == 'C01':
            # Fit I = a/cos(θ) + b
            def fit_func(theta, a, b):
                return a / np.cos(np.radians(theta)) + b

            initial_guess = [0.3, 0.1]

        elif channel in ['C02', 'C05']:
            # Fit I = a*sin(2θ) + b
            def fit_func(theta, a, b):
                return a * np.sin(2 * np.radians(theta)) + b

            initial_guess = [0.3, 0.4]

        elif channel in ['C03', 'C06']:
            # Fit I = a*exp(-(θ-θ0)²/2σ²)/cos(θ) + b
            def fit_func(theta, a, theta0, sigma, b):
                theta_rad = np.radians(theta)
                theta0_rad = np.radians(theta0)
                gaussian = np.exp(-(theta_rad - theta0_rad) ** 2 / (2 * np.radians(sigma) ** 2))
                return a * gaussian / np.cos(theta_rad) + b

            initial_guess = [0.5, 45, 20, 0.3]

        elif channel == 'C04':
            # Fit I = a + b (constant)
            def fit_func(theta, a, b):
                return a * np.ones_like(theta) + b

            initial_guess = [0.02, 0.01]

        try:
            popt, pcov = curve_fit(fit_func, theta_data, intensity_data, p0=initial_guess)
            return popt, np.sqrt(np.diag(pcov))
        except:
            return initial_guess, np.ones_like(initial_guess) * 0.1

    def fit_azimuth_function(self, phi_data, intensity_data):
        """Fit azimuth angle dependence to data"""

        def fit_func(phi, g, f_amp, f_width, b_amp, b_width, norm):
            phi_rad = np.radians(phi)

            # Henyey-Greenberg base
            hg = (1 - g ** 2) / ((1 + g ** 2 - 2 * g * np.cos(phi_rad)) ** (3 / 2))

            # Forward peak
            forward = f_amp * np.exp(-phi_rad ** 2 / (2 * np.radians(f_width) ** 2))

            # Backward peak
            back_phi = np.abs(phi_rad - np.pi)
            backward = b_amp * np.exp(-back_phi ** 2 / (2 * np.radians(b_width) ** 2))

            return (hg + forward + backward) * norm

        initial_guess = [0.85, 0.05, 30, 0.02, 45, 0.1]

        try:
            popt, pcov = curve_fit(fit_func, phi_data, intensity_data, p0=initial_guess)
            return popt, np.sqrt(np.diag(pcov))
        except:
            return initial_guess, np.ones_like(initial_guess) * 0.1

    def create_analytical_lut(self, cod_values, sza_values, channels):
        """Create analytical lookup table"""

        lut = {}

        for channel in channels:
            lut[channel] = {}

            for cod in cod_values:
                lut[channel][cod] = {}

                for sza in sza_values:
                    # These parameters would be fitted from your Monte Carlo data
                    # Here I provide typical values based on the patterns in your figures

                    if channel == 'C01':
                        zenith_params = {'a': 0.3 + 0.02 * cod, 'b': 0.1}
                    elif channel in ['C02', 'C05']:
                        zenith_params = {'a': 0.2 + 0.01 * cod, 'b': 0.35 + 0.005 * cod}
                    elif channel in ['C03', 'C06']:
                        zenith_params = {'a': 0.4 + 0.01 * cod, 'theta0': 50, 'sigma': 25, 'b': 0.3}
                    elif channel == 'C04':
                        zenith_params = {'a': 0.01 + 0.001 * cod, 'b': 0.005}

                    # Azimuth parameters (less dependent on COD for water clouds)
                    azimuth_params = {
                        'g': 0.85 + 0.005 * np.log(cod + 1),  # Slight COD dependence
                        'f_amp': 0.05,
                        'f_width': 30,
                        'b_amp': 0.02,
                        'b_width': 45,
                        'norm': 0.1 + 0.01 * np.sin(np.radians(sza))
                    }

                    lut[channel][cod][sza] = {
                        'zenith_params': zenith_params,
                        'azimuth_params': azimuth_params
                    }

        return lut

    def evaluate_adf(self, theta, phi, channel, cod, sza, lut):
        """Evaluate analytical ADF for given geometry"""

        params = lut[channel][cod][sza]

        return self.combined_angular_function(
            theta, phi, channel,
            params['zenith_params'],
            params['azimuth_params']
        )


# Example usage and validation
def example_usage():
    """Example of how to use the analytical ADF"""

    adf = AnalyticalADF()

    # Create example data grids
    theta = np.arange(0, 90, 5)
    phi = np.arange(-180, 180, 10)
    channels = ['C01', 'C02', 'C03', 'C04', 'C05', 'C06']
    cod_values = [1, 5, 10, 20, 30]
    sza_values = [0, 15, 30, 45, 60]

    # Create analytical LUT
    lut = adf.create_analytical_lut(cod_values, sza_values, channels)

    # Example: Evaluate for specific geometry
    theta_eval, phi_eval = np.meshgrid(theta, phi)

    intensity = adf.evaluate_adf(
        theta_eval, phi_eval, 'C03', cod=10, sza=30, lut=lut
    )

    print(f"Analytical ADF evaluation completed")
    print(f"Intensity shape: {intensity.shape}")
    print(f"Intensity range: {intensity.min():.4f} - {intensity.max():.4f}")

    return adf, lut


if __name__ == "__main__":
    adf, lut = example_usage()