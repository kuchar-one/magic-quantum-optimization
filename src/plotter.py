import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors, cm, gridspec
from matplotlib.colorbar import ColorbarBase
from qutip import *
import sys
import math

# Configure LaTeX with minimal style changes
mpl.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"""
        \usepackage{amsmath}
        \usepackage{physics}
        \usepackage{braket}
    """,
    "axes.titlesize": 14  # Slightly larger titles (default is 12)
})
# Configure matplotlib to use LaTeX for rendering text and a serif font family
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

from qutip_quantum_ops import gkp_operator_new, operator_new

class PlateauTwoSlopeNorm(colors.TwoSlopeNorm):
    def __init__(self, vcenter, plateau_size, vmin=None, vmax=None):
        super().__init__(vcenter=vcenter, vmin=vmin, vmax=vmax)
        self.plateau_size = plateau_size
        
    def __call__(self, value, clip=None):
        result, is_scalar = self.process_value(value)
        self.autoscale_None(result)
        
        if not self.vmin <= self.vcenter <= self.vmax:
            raise ValueError("vmin, vcenter, vmax must increase monotonically")
            
        plateau_lower = self.vcenter - self.plateau_size/2
        plateau_upper = self.vcenter + self.plateau_size/2
        
        x_points = [self.vmin, plateau_lower, plateau_upper, self.vmax]
        y_points = [0, 0.5, 0.5, 1]
        
        result = np.ma.masked_array(
            np.interp(result, x_points, y_points, left=-np.inf, right=np.inf),
            mask=np.ma.getmask(result))
            
        if is_scalar:
            result = np.atleast_1d(result)[0]
        return result
        
    def inverse(self, value):
        if not self.scaled():
            raise ValueError("Not invertible until both vmin and vmax are set")
            
        plateau_lower = self.vcenter - self.plateau_size/2
        plateau_upper = self.vcenter + self.plateau_size/2
        
        x_points = [0, 0.5, 0.5, 1]
        y_points = [self.vmin, plateau_lower, plateau_upper, self.vmax]
        
        return np.interp(value, x_points, y_points, left=-np.inf, right=np.inf)

def plot_states(states, titles, xvec=None, yvec=None, 
                cmap='inferno', vmin=-0.23, vmax=0.23, vcenter=0, 
                plateau_size=0.03, figsize=(12, 7.2)):
    """Plot multiple states in subplots with shared colorbar."""
    if xvec is None:
        xvec = np.linspace(-5, 5, 1000)
    if yvec is None:
        yvec = np.linspace(-5, 5, 1000)
    
    num_states = len(states)
    rows = math.ceil(num_states / 3)
    cols = 3

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(rows, cols + 1, figure=fig, 
                         width_ratios=[1]*cols + [0.05])
    
    norm = PlateauTwoSlopeNorm(vcenter=vcenter, plateau_size=plateau_size, 
                             vmin=vmin, vmax=vmax)
    
    axes = []
    for i in range(num_states):
        row = i // cols
        col = i % cols
        ax = fig.add_subplot(gs[row, col])
        axes.append(ax)
        
        W = wigner(states[i], xvec, yvec)
        ax.contourf(xvec, yvec, W, 1000, cmap=cmap, norm=norm, zorder=-1)
        ax.grid(False)
        ax.set_xlabel('$x$', fontsize=10)
        ax.set_ylabel('$p$', fontsize=10)
        ax.set_title(titles[i], pad=12)  # Increased title padding
        ax.set_rasterization_zorder(0)
    
    cax = fig.add_subplot(gs[:, -1])
    cbar = ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')
    plt.tight_layout()
    return fig, axes, cbar

def plot_single_state(state, title, xvec=None, yvec=None,
                      cmap='inferno', vmin=-0.23, vmax=0.23, vcenter=0,
                      plateau_size=0.03, figsize=(5, 4)):
    """Plot a single state with dedicated colorbar."""
    if xvec is None:
        xvec = np.linspace(-5, 5, 1000)
    if yvec is None:
        yvec = np.linspace(-5, 5, 1000)

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 0.05], figure=fig)
    
    norm = PlateauTwoSlopeNorm(vcenter=vcenter, plateau_size=plateau_size,
                             vmin=vmin, vmax=vmax)
    
    ax = fig.add_subplot(gs[0, 0])
    W = wigner(state, xvec, yvec)
    ax.contourf(xvec, yvec, W, 1000, cmap=cmap, norm=norm, zorder=-1)
    ax.grid(False)
    ax.set_xlabel('$x$', fontsize=10)
    ax.set_ylabel('$p$', fontsize=10)
    ax.set_title(title, pad=12)  # Increased title padding
    ax.set_rasterization_zorder(0)
    
    cax = fig.add_subplot(gs[0, 1])
    ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')
    plt.tight_layout()
    return fig, ax, cax

def plot_single_state_bare(state, xvec=None, yvec=None,
                      cmap='inferno', vmin=-0.23, vmax=0.23, vcenter=0,
                      plateau_size=0.03, figsize=(4, 3)):
    """Plot a single state with dedicated colorbar."""
    if xvec is None:
        xvec = np.linspace(-6, 6, 1000)
    if yvec is None:
        yvec = np.linspace(-6, 6, 1000)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    
    norm = PlateauTwoSlopeNorm(vcenter=vcenter, plateau_size=plateau_size,
                             vmin=vmin, vmax=vmax)
    
    W = wigner(state, xvec, yvec)
    ax.contourf(xvec, yvec, W, 1000, cmap=cmap, norm=norm, zorder=-1)
    ax.set_rasterization_zorder(0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    return fig, ax

def plot_wigner_with_marginals(state, xvec=None, yvec=None,
                                cmap='inferno', vmin=-0.23, vmax=0.23, vcenter=0,
                                plateau_size=0.01, figsize=(6, 6),
                                margin_frac=0.15):
    """
    Plot Wigner function of a state with its x- and p-quadrature marginals.

    Parameters
    ----------
    state : quantum state
        The state for which to compute the Wigner function.
    xvec, yvec : array-like, optional
        Grid vectors for x (position) and p (momentum).
        Defaults to 1000 points each between -6 and 6.
    cmap : str
        Colormap for the Wigner contour.
    vmin, vmax, vcenter, plateau_size : float
        Parameters for the PlateauTwoSlopeNorm color normalization.
    figsize : tuple
        Size of the overall figure.
    margin_frac : float
        Fraction of figure reserved for marginal plots.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax_wigner, ax_x, ax_p : matplotlib.axes.Axes
        Axes for the Wigner plot, x-marginal, and p-marginal.
    """
    # Set up default grids
    if xvec is None:
        xvec = np.linspace(-6, 6, 300)
    if yvec is None:
        yvec = np.linspace(-6, 6, 300)

    # Compute Wigner function
    W = wigner(state, xvec, yvec)

    # Compute marginals: integrate W over p (yvec) for x-dist, over x (xvec) for p-dist
    dx = xvec[1] - xvec[0]
    dy = yvec[1] - yvec[0]
    px = np.trapz(W, yvec, axis=0)
    pp = np.trapz(W, xvec, axis=1)

    # Normalize marginals to unity
    px /= np.trapz(px, xvec)
    pp /= np.trapz(pp, yvec)

    # Layout: main Wigner + top x-marginal + right p-marginal
    fig = plt.figure(figsize=figsize)
    # Margins sizes in figure fraction
    m = margin_frac
    # Main Wigner axes
    ax_wigner = fig.add_axes([m, m, 1-2*m, 1-2*m])
    # Top marginal (x)
    ax_x = fig.add_axes([m, 1-m/2, 1-2*m, m/2], sharex=ax_wigner)
    # Right marginal (p)
    ax_p = fig.add_axes([1-m/2, m, m/2, 1-2*m], sharey=ax_wigner)

    # Color normalization
    norm = PlateauTwoSlopeNorm(vcenter=vcenter, plateau_size=plateau_size,
                                vmin=vmin, vmax=vmax)

    # Plot Wigner contour
    cf = ax_wigner.contourf(xvec, yvec, W, 200, cmap=cmap, norm=norm)

    # Plot marginals
    ax_x.plot(xvec, px)
    ax_x.set_ylabel('P(x)')

    ax_p.plot(pp, yvec)
    ax_p.set_xlabel('P(p)')

    # Return
    return fig, ax_wigner, ax_x, ax_p


# Example usage with physics notation
if __name__ == "__main__":
    # Generate sample states
    nvals = [3, 5, 12, 30, 80, 150]
    eigvals_states = [operator_new(N, 3, 0, 10, 100).groundstate() for N in nvals]
    print('test')
    states = [es[1] for es in eigvals_states]
    titles = [f"$\\textbf{{({chr(97+i)})}}$ $N={n}$" for i, n in enumerate(nvals)]  # Using ket notation with letters

    # Plot multiple states
    fig_multi, axes_multi, cbar_multi = plot_states(states, titles)
    plt.savefig('multiple_states.pdf', bbox_inches='tight', dpi=600)