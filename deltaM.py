import numpy as np
import matplotlib.pyplot as plt
from IPython.lib.deepreload import original_import
from scipy.special import legendre
from LBL_funcs_fullSpectrum import *

class DeltaMApproximation:
    """
    Implementation of the delta-M approximation for radiative transfer

    The delta-M method addresses the computational challenges of highly
    forward-peaked scattering phase functions by:
    1. Truncating the phase function expansion at order N
    2. Scaling the optical properties to account for truncated forward scattering
    3. Adding a delta function to preserve the forward scattering peak
    """

    def __init__(self, truncation_order=64,n_streams=200):
        """
        Initialize delta-M approximation

        Parameters:
        -----------
        n_streams : int
            Number of streams for discrete ordinates
        truncation_order : int
            Order at which to truncate Legendre expansion
        """

        self.N = truncation_order
        self.n_streams = n_streams

    def compute_legendre_moments(self, theta, phase_function):
        """
        Compute Legendre moments of phase function

        Parameters:
        -----------
        theta : array
            Scattering angles (radians)
        phase_function : array
            Phase function values P(theta)

        Returns:
        --------
        chi : array
            Legendre moments chi_l
        """
        n_moments = self.n_streams
        chi = np.zeros(n_moments)
        cos_theta = np.cos(theta)
        # Compute moments using numerical integration
        for l in range(self.N + 2):  # When use in CODE, we use self.N + 2
            P_l = legendre(l) # Legendre polynomial of order l
            # Integrate P(theta) * P_l(cos(theta)) * sin(theta) dtheta
            integrand = phase_function * P_l(cos_theta) * np.sin(theta)
            chi[l] =  np.trapz(integrand, theta) / 2.0  # Lin et al. (2018) eq 4, wiscombe 1977 eq(4)
        #chi = chi / chi[0]  # Normalize to chi_0 = 1
        return chi

    def mie_phase_function(self, g):
        """
        Approximate Mie phase function (simplified for demonstration)
        In practice, you would use full Mie theory calculations

        Parameters:
        -----------
        theta : array
            Scattering angles
        size_parameter : float
            Mie size parameter (2*pi*r/lambda)
        refractive_index : complex
            Complex refractive index

        Returns:
        --------
        phase_function : array
            Phase function values
        """
        angles = np.linspace(0, 180, 361)  # degrees
        theta = np.deg2rad(angles)
        mu = np.cos(theta)

        lam = 0.4
        r = 10
        data_w = np.genfromtxt('data/profiles/water_refraction.csv', delimiter=',')
        real_w = np.interp(lam, data_w[:, 0], data_w[:, 1])
        img_w = np.interp(lam, data_w[:, 0], data_w[:, 2])
        refrac_w = real_w + 1j * img_w
        phase_Mie = phaseFunction(lam, r, refrac_w, mu, theta)

        return phase_Mie, theta, mu
        #mu = np.cos(theta)
        #return (1 - g ** 2) / (2 * (1 + g ** 2 - 2 * g * mu) ** 1.5)

    def apply_delta_m_scaling(self, chi, tau, omega):
        """
        Apply delta-M scaling to optical properties

        Parameters:
        -----------
        tau : float
            Original optical depth
        omega : float
            Original single scattering albedo
        chi : array
            Legendre moments of phase function

        Returns:
        --------
        tau_scaled : float
            Scaled optical depth
        omega_scaled : float
            Scaled single scattering albedo
        chi_scaled : array
            Scaled Legendre moments
        f : float
            Forward scattering fraction
        """
        # Forward scattering fraction (fraction of scattering in exact forward direction)
        f = chi[self.N] # Lin et al. (2018) eq(9)

        # Scaled optical depth
        tau_scaled = tau * (1 - omega * f)
        # Scaled single scattering albedo
        if tau_scaled > 0:
            omega_scaled = omega * (1 - f) / (1 - omega * f)
        else:
            omega_scaled = omega

        # Scaled phase function moments
        chi_scaled = np.zeros_like(chi)
        for l in range(len(chi)):
            if l <= self.N:
                chi_scaled[l] = (chi[l] - f) / (1 - f)
            else:
                chi_scaled[l] = 0  # Truncated
        #chi_scaled = chi_scaled/ chi_scaled[0]  # Normalize to chi_0 = 1
        return tau_scaled, omega_scaled, chi_scaled, f

    def truncated_phase_function(self, chi_scaled, theta, f=0):
        """
        Reconstruct phase function from scaled Legendre moments
        phase = (2*l+1) * chi_l * P_l(cos(theta))  eq(3) Wiscombe,1977

        Parameters:
        -----------
        chi_scaled : array
            Scaled Legendre moments
        theta : array
            Scattering angles
        f : float
            Forward scattering fraction for delta function

        Returns:
        --------
        phase_reconstructed : array
            Reconstructed phase function
        """
        cos_theta = np.cos(theta)
        phase_reconstructed = np.zeros_like(theta)

        # Sum truncated Legendre series (include (2l+1)/2 factor!)
        M_tunrc = min(len(chi_scaled), min(len(chi_scaled), self.N))
        for l in range(M_tunrc):
            P_l = legendre(l)
            phase_reconstructed += (2 * l + 1) * chi_scaled[l] * P_l(cos_theta) # eq(3) Wiscombe,1977

        # in Monte Carlo, we do not need to add delta function explicitly
        # delta_contribution = 0
        # if f > 0:
        #     # dirac delta contribution
        #     for l in range(M_tunrc, self.n_streams + 1):
        #         delta_contribution += (2 * l + 1) * f * legendre(l)(cos_theta)
        #     phase_reconstructed += delta_contribution
        return phase_reconstructed

    def demonstrate_delta_m(self, original_phase, theta, g = 0.85):
        """
        Demonstrate the delta-M approximation process
        """
        # Compute Legendre moments
        chi_original = self.compute_legendre_moments(theta, original_phase)
        # Apply delta-M scaling
        tau=1.0
        omega=0.95
        tau_scaled, omega_scaled, chi_scaled, f = self.apply_delta_m_scaling(chi_original,tau, omega)
        # Reconstruct scaled phase function
        truncated_phase = self.truncated_phase_function(chi_scaled, theta, f)
        truncated_phase = np.clip(truncated_phase, 0,  None)
        #norm = np.trapz(truncated_phase * np.sin(theta), theta) * 2 * np.pi
        #truncated_phase /= norm
        trunc_g = chi_scaled/chi_scaled[0]

        return {
            'theta': theta,
            'original_phase': original_phase,
            'truncated_phase':truncated_phase,
            'chi_original': chi_original,
            'chi_scaled': chi_scaled,
            'tau_original': tau,
            'tau_scaled': tau_scaled,
            'omega_original': omega,
            'omega_scaled': omega_scaled,
            'f_trunc': f,
            'trunc_g':trunc_g[1],
        }

def plot_delta_m_results(results,truncation_order,n_streams):
    """
    Plot the results of delta-M approximation
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    theta_deg = np.degrees(results['theta'])
    # Phase functions comparison
    ax1.semilogy(theta_deg, results['original_phase'], 'k-', linewidth=2, label='Original Mie')
    ax1.semilogy(theta_deg, results['truncated_phase'], 'r--', linewidth = 2, label = 'Truncated')
    ax1.set_xlabel('Scattering Angle (degrees)')
    ax1.set_ylabel('Phase Function')
    ax1.set_title('Phase Function Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 180)
    # Forward scattering detail
    forward_mask = theta_deg <= 30
    ax2.semilogy(theta_deg[forward_mask], results['original_phase'][forward_mask], 'k-', linewidth=2, label='Original')
    ax2.semilogy(theta_deg[forward_mask], results['truncated_phase'][forward_mask], 'b-', linewidth=2, label='Delta-M')
    ax2.set_xlabel('Scattering Angle (degrees)')
    ax2.set_ylabel('Phase Function')
    ax2.set_title('Forward Scattering Detail')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Legendre moments
    l_values = np.arange(len(results['chi_original']))
    ax3.plot(l_values[:n_streams], np.abs(results['chi_original'][:n_streams]), 'ko-', markersize=4, label='Original')
    ax3.plot(l_values[:n_streams], np.abs(results['chi_scaled'][:n_streams]), 'bo-', markersize=4, label='Scaled')
    ax3.axvline(truncation_order, color='r', linestyle='--', alpha=0.7, label='Truncation order')
    ax3.set_xlabel('Legendre Order l')
    ax3.set_ylabel('Legendre Coefficients')
    ax3.set_title('Legendre Moments')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Optical properties summary
    ax4.axis('off')
    props_text = f"""
    Optical Properties Summary:

    Original:
    • Optical depth: {results['tau_original']:.3f}
    • Single scattering albedo: {results['omega_original']:.3f}

    Delta-M Scaled:
    • Optical depth: {results['tau_scaled']:.3f} 
    • Single scattering albedo: {results['omega_scaled']:.3f}
    • Forward fraction: {results['forward_fraction']:.3f}

    Key Benefits:
    • Removes spurious oscillations in truncated phase function
    • Preserves forward scattering through delta function
    • Maintains energy conservation 
    • Improves computational efficiency
    """
    ax4.text(0.05, 0.95, props_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    # Initialize delta-M approximation
    truncation_order = 36 #16,32,64
    n_streams = 68
    delta_m = DeltaMApproximation(truncation_order,n_streams)
    g = 0.9
    original_phase, theta, mu = delta_m.mie_phase_function(g)
    # Demonstrate for a water droplet (large size parameter)
    print("Demonstrating Delta-M approximation for large water droplet...")

    results = delta_m.demonstrate_delta_m(original_phase, theta, g)

    print(f"\nResults:")
    print(f"Forward scattering fraction: {results['forward_fraction']:.4f}")
    print(f"Original optical depth: {results['tau_original']:.3f}")
    print(f"Scaled optical depth: {results['tau_scaled']:.3f}")
    print(f"Original single scattering albedo: {results['omega_original']:.3f}")
    print(f"Scaled single scattering albedo: {results['omega_scaled']:.3f}")

    # Plot results
    plot_delta_m_results(results, truncation_order, n_streams)