import qutip as qt
import matplotlib.pyplot as plt
import numpy as np
import os
from mpmath import sqrt, pi, jtheta, exp, fac
from scipy.optimize import minimize_scalar
from scipy.special import hermite, factorial, gamma
from typing import Tuple, Literal, Union, NoReturn


def high_dim_generator(u, phi, c, k):
    """
    Generate and save the high dimensional operator O_2(N, a, phi, c, k) = (position(N)^2 - a^2 * qeye(N))^2 + c * a * 4^k * (fac(k))^2 / (pi * fac(2 * k)) * (-1/4)^k * (expm(-1j * (momentum(N) * a + phi/2)) - expm(1j * (momentum(N) * a + phi/2)))^(2 * k).
    
    Parameters
    ----------
    a : float
        A parameter of the operator.
    phi : float
        A parameter of the operator.
    c : float
        A parameter of the operator.
    k : int
        A parameter of the operator.

    Returns
    -------
    Qobj
        The operator O_2(N, a, phi, c, k).
    """

    dim = 2500
    plus = 1j * (qt.momentum(dim) * u + phi / 2)
    minus = -1j * (qt.momentum(dim) * u + phi / 2)

    high_dim = (qt.position(dim) ** 2 - u**2 * qt.qeye(dim)) ** 2 + c * u * gamma(
        k + 1
    ) / (np.sqrt(np.pi) * gamma(k + 0.5)) * (-1 / 4) ** k * (
        minus.expm() - plus.expm()
    ) ** (
        2 * k
    )

    print(
        f"Generating pre-truncation form of the SQE Operator for u = {u}, phi = {phi}, c = {c}, k = {k}..."
    )

    os.makedirs("cache/operators", exist_ok=True)
    qt.qsave(
        high_dim, f"cache/operators/high_dim_u{u:.2f}_phi{phi:.2f}_c{c:.2f}_k{k:.2f}"
    )

    return high_dim


def operator_new(N, u, phi, c, k):
    """
    Load or generate a high-dimensional operator and return its truncated version.

    This function attempts to load a previously saved high-dimensional operator
    from a file. If the file does not exist, it generates the operator using the
    provided parameters and saves it. The operator is then truncated to the specified
    dimension N and returned.

    Parameters
    ----------
    N : int
        The dimension to which the operator should be truncated.
    a : float
        A parameter used in the generation of the operator.
    phi : float
        A parameter used in the generation of the operator.
    c : float
        A parameter used in the generation of the operator.
    k : int
        A parameter used in the generation of the operator.

    Returns
    -------
    Qobj
        The truncated high-dimensional operator.
    """
    filename = f"cache/operators/high_dim_u{u:.2f}_phi{phi:.2f}_c{c:.2f}_k{k:.2f}"

    if os.path.isfile(f"{filename}.qu"):
        high_dim = qt.qload(filename)
    else:
        high_dim = high_dim_generator(u, phi, c, k)

    return qt.Qobj(high_dim.full()[:N, :N])


def high_dim_gkp_generator():
    """
    Generate and save a high-dimensional GKP squeezing operator.

    The operator is calculated in a 5000-dimensional Fock space and then saved to a file.

    Returns
    -------
    oper
        The high-dimensional gkp squeezing operator.
    """
    dim = 5000
    sin_term1 = +1j * qt.position(dim) * np.sqrt(np.pi) / 2
    sin_term2 = +1j * qt.momentum(dim) * np.sqrt(np.pi)
    high_dim = 2 * operator_sin(sin_term1) ** 2 + 2 * operator_sin(sin_term2) ** 2

    print(f"Generating pre-truncation form of the GKP Operator...")

    qt.qsave(high_dim, f"cache/operators/high_dim_gkp")

    return high_dim


def gkp_operator_new(N):
    """
    Load or generate a high-dimensional operator and return its truncated version.

    This function attempts to load a previously saved high-dimensional operator
    from a file. If the file does not exist, it generates the operator using the
    provided parameters and saves it. The operator is then truncated to the specified
    dimension N and returned.

    Parameters
    ----------
    N : int
        The dimension to which the operator should be truncated.

    Returns
    -------
    Qobj
        The truncated high-dimensional operator.
    """
    filename = f"cache/operators/high_dim_gkp"

    if os.path.isfile(f"{filename}.qu"):
        high_dim = qt.qload(filename)
    else:
        high_dim = high_dim_gkp_generator()

    return qt.Qobj(high_dim.full()[:N, :N])


def gaussian_limit(u, c):
    return min(
        [
            u * c / np.pi,
            minimize_scalar(
                lambda r: float(
                    u**4
                    + 3 / (4 * exp(4 * r))
                    - u**2 / exp(2 * r)
                    + u * c / pi * jtheta(3, -1 / 2 * pi, exp(-(u**2 * exp(2 * r))))
                ),
                bounds=[-10, 10],
                method="bounded",
            ).fun,
        ]
    )


def beam_splitter(N, one_in, two_in, theta=np.pi / 4):
    """Optimized beam splitter interaction with caching

    Args:
        N (int): truncated Fock space dimension
        one_in (ket or oper): first mode input
        two_in (ket or oper): second mode input
        theta (float, optional): BS parameter, balanced default

    Returns:
        oper: two-mode output density matrix
    """
    # Convert inputs to density matrices if needed
    one_dm = qt.ket2dm(one_in) if one_in.type == "ket" else one_in
    two_dm = qt.ket2dm(two_in) if two_in.type == "ket" else two_in

    # Cache filename
    filename = f"cache/operators/beam_splitter_qutip_N{N}_theta{theta:.2f}"

    # Check if cached unitary exists
    if os.path.isfile(f"{filename}.qu"):
        unitary = qt.qload(filename)
    else:
        # Create destruction operators using QutIP's tensor
        destroy_one = qt.tensor(qt.destroy(N), qt.identity(N))
        destroy_two = qt.tensor(qt.identity(N), qt.destroy(N))

        # Compute the unitary operator
        generator = -theta * (
            destroy_one.dag() * destroy_two - destroy_two.dag() * destroy_one
        )
        unitary = generator.expm()

        # Save the unitary to cache
        os.makedirs("cache/operators", exist_ok=True)
        qt.qsave(unitary, filename)

    # Compute final state
    input_dm = qt.tensor(one_dm, two_dm)
    return unitary.dag() * input_dm * unitary


def measure_mode(N, two_mode_dm, projector, mode):
    """Optimized projection measurement of two mode state

    Args:
        N (int): truncated Fock space dimension
        two_mode_dm (oper): two mode state density operator
        projector (oper): measurement projection
        mode (1 or 2): mode number

    Returns:
        oper: conditional output density matrix in untouched mode
    """
    id_N = qt.identity(N)

    # Pre-compute measurement operator
    measurement = (
        qt.tensor(projector, id_N) if mode == 1 else qt.tensor(id_N, projector)
    )

    # Perform measurement and partial trace
    measured_state = two_mode_dm * measurement
    traced_state = measured_state.ptrace(1 if mode == 1 else 0)

    return traced_state.unit()


def breeding(N, rounds, input_state, projector):
    """Breeding protocol for generating a GKP state.

    Args:
        N (int): truncated Fock space dimension.
        rounds (int): number of breeding rounds.
        input_state (oper): input state.
        projector (oper): measurement projector.

    Returns:
        oper: output density matrix of the protocol.
    """
    if rounds == 0:
        return input_state

    else:
        temp = beam_splitter(N, input_state, input_state)
        new = measure_mode(N, temp, projector, 1)
        output_state = breeding(N, rounds - 1, new, projector)

    return output_state


def squeezed_cat(N, a, r):
    """
    Generate a squeezed Schrödinger cat state.

    Parameters
    ----------
    N : int
        The dimension of the Fock space.
    a : float
        The displacement amplitude.
    r : float
        The squeezing parameter.

    Returns
    -------
    Qobj
        A normalized squeezed Schrödinger cat state.
    """
    plus = qt.displace(N, a / np.sqrt(2)) * qt.squeeze(N, r) * qt.basis(N, 0)
    minus = qt.displace(N, -a / np.sqrt(2)) * qt.squeeze(N, r) * qt.basis(N, 0)
    return (plus + minus) / (plus + minus).norm()


def beam_splitter(
    N: int, one_in: qt.Qobj, two_in: qt.Qobj, theta: float = np.pi / 4
) -> qt.Qobj:
    """Optimized beam splitter interaction with caching

    Args:
        N (int): truncated Fock space dimension
        one_in (ket or oper): first mode input
        two_in (ket or oper): second mode input
        theta (float, optional): BS parameter, balanced default

    Returns:
        oper: two-mode output density matrix
    """
    # Convert inputs to density matrices if needed
    one_dm = qt.ket2dm(one_in) if one_in.type == "ket" else one_in
    two_dm = qt.ket2dm(two_in) if two_in.type == "ket" else two_in

    # Cache filename
    filename = f"cache/operators/beam_splitter_qutip_N{N}_theta{theta:.2f}"

    # Check if cached unitary exists
    if os.path.isfile(f"{filename}.qu"):
        unitary = qt.qload(filename)
    else:
        # Create destruction operators using QutIP's tensor
        destroy_one = qt.tensor(qt.destroy(N), qt.identity(N))
        destroy_two = qt.tensor(qt.identity(N), qt.destroy(N))

        # Compute the unitary operator
        generator = -theta * (
            destroy_one.dag() * destroy_two - destroy_two.dag() * destroy_one
        )
        unitary = generator.expm()

        # Save the unitary to cache
        os.makedirs("cache/operators", exist_ok=True)
        qt.qsave(unitary, filename)

    # Compute final state
    input_dm = qt.tensor(one_dm, two_dm)
    return unitary.dag() * input_dm * unitary


def measure_mode(
    N: int, two_mode_dm: qt.Qobj, projector: qt.Qobj, mode: Literal[1, 2]
) -> qt.Qobj:
    """Optimized projection measurement of two mode state

    Args:
        N (int): truncated Fock space dimension
        two_mode_dm (oper): two mode state density operator
        projector (oper): measurement projection
        mode (1 or 2): mode number

    Returns:
        oper: conditional output density matrix in untouched mode
    """
    id_N = qt.identity(N)

    # Pre-compute measurement operator
    measurement = (
        qt.tensor(projector, id_N) if mode == 1 else qt.tensor(id_N, projector)
    )

    # Perform measurement and partial trace
    measured_state = two_mode_dm * measurement
    traced_state = measured_state.ptrace(1 if mode == 1 else 0)

    return traced_state.unit()


SQRT_PI = np.sqrt(np.pi)


def get_XYZU_cos(N: int) -> Tuple[qt.Qobj, qt.Qobj, qt.Qobj, qt.Qobj]:
    cache_dir = "cache/operators"
    os.makedirs(cache_dir, exist_ok=True)
    X_path = f"{cache_dir}/X_cos_{N}"
    Y_path = f"{cache_dir}/Y_cos_{N}"
    Z_path = f"{cache_dir}/Z_cos_{N}"
    U_path = f"{cache_dir}/U_cos_{N}"

    if os.path.isfile(X_path + ".qu"):
        X = qt.qload(X_path)
    else:
        # (I - cos(p * sqrt(pi))) / 2  ==  sin^2(p * sqrt(pi) / 2)
        X = (qt.qeye(N) - (qt.momentum(N) * SQRT_PI).cosm()) / 2
        qt.qsave(X, X_path)

    if os.path.isfile(Y_path + ".qu"):
        Y = qt.qload(Y_path)
    else:
        # FIXED: use (x+p)*sqrt(pi) (not 2*sqrt(pi)*(x+p))
        Y = (qt.qeye(N) - ((qt.position(N) + qt.momentum(N)) * SQRT_PI).cosm()) / 2
        qt.qsave(Y, Y_path)

    if os.path.isfile(Z_path + ".qu"):
        Z = qt.qload(Z_path)
    else:
        Z = (qt.qeye(N) - (qt.position(N) * SQRT_PI).cosm()) / 2
        qt.qsave(Z, Z_path)

    if os.path.isfile(U_path + ".qu"):
        U = qt.qload(U_path)
    else:
        # U = 1/2 * ( sin^2(x*sqrt(pi)) + sin^2(p*sqrt(pi)) )
        # write as combination with cos(2θ) for clarity:
        U = 0.25 * (
            2 * qt.qeye(N)
            - (2 * qt.position(N) * SQRT_PI).cosm()
            - (2 * qt.momentum(N) * SQRT_PI).cosm()
        )
        qt.qsave(U, U_path)

    return X, Y, Z, U


def get_XYZU_sin(N: int) -> Tuple[qt.Qobj, qt.Qobj, qt.Qobj, qt.Qobj]:
    cache_dir = "cache/operators"
    os.makedirs(cache_dir, exist_ok=True)
    X_path = f"{cache_dir}/X_sin_{N}"
    Y_path = f"{cache_dir}/Y_sin_{N}"
    Z_path = f"{cache_dir}/Z_sin_{N}"
    U_path = f"{cache_dir}/U_sin_{N}"

    if os.path.isfile(X_path + ".qu"):
        X = qt.qload(X_path)
    else:
        # sin^2(p*sqrt(pi)/2)
        X = ((qt.momentum(N) * SQRT_PI / 2).sinm()) ** 2
        qt.qsave(X, X_path)

    if os.path.isfile(Y_path + ".qu"):
        Y = qt.qload(Y_path)
    else:
        # sin^2((x+p)*sqrt(pi)/2)
        Y = ((SQRT_PI * (qt.position(N) + qt.momentum(N)) / 2).sinm()) ** 2
        qt.qsave(Y, Y_path)

    if os.path.isfile(Z_path + ".qu"):
        Z = qt.qload(Z_path)
    else:
        Z = ((qt.position(N) * SQRT_PI / 2).sinm()) ** 2
        qt.qsave(Z, Z_path)

    if os.path.isfile(U_path + ".qu"):
        U = qt.qload(U_path)
    else:
        # 1/2 * ( sin^2(x*sqrt(pi)) + sin^2(p*sqrt(pi)) )
        U = 0.5 * (
            (qt.position(N) * SQRT_PI).sinm() ** 2
            + (qt.momentum(N) * SQRT_PI).sinm() ** 2
        )
        qt.qsave(U, U_path)

    return X, Y, Z, U


def get_XYZU_paper(N: int) -> Tuple[qt.Qobj, qt.Qobj, qt.Qobj, qt.Qobj]:
    cache_dir = "cache/operators"
    os.makedirs(cache_dir, exist_ok=True)
    X_path = f"{cache_dir}/X_paper_{N}"
    Y_path = f"{cache_dir}/Y_paper_{N}"
    Z_path = f"{cache_dir}/Z_paper_{N}"
    U_path = f"{cache_dir}/U_paper_{N}"

    if os.path.isfile(X_path + ".qu"):
        X = qt.qload(X_path)
    else:
        # cos(p*sqrt(pi))
        X = ((qt.momentum(N) * SQRT_PI).cosm())
        qt.qsave(X, X_path)

    if os.path.isfile(Z_path + ".qu"):
        Z = qt.qload(Z_path)
    else:
        # cos(x*sqrt(pi))
        Z = ((qt.position(N) * SQRT_PI).cosm())
        qt.qsave(Z, Z_path)

    if os.path.isfile(Y_path + ".qu"):
        Y = qt.qload(Y_path)
    else:
        # i*X*Z
        Y = 1j*X*Z
        qt.qsave(Y, Y_path)

    if os.path.isfile(U_path + ".qu"):
        U = qt.qload(U_path)
    else:
        U = 2*qt.qeye(N) - (X**2 + Z**2 + Y**2)/3
        qt.qsave(U, U_path)

    return X, Y, Z, U


def high_dim_magic_generator_cos(A: float, B: float, C: float) -> qt.Qobj:
    N = 1000
    X, Y, Z, U = get_XYZU_cos(N)
    print(f"Generating pre-truncation form of the GKP Operator...")
    high_dim = A * X + B * Y + C * Z + np.sqrt(A**2 + B**2 + C**2) * U
    qt.qsave(high_dim, f"cache/operators/high_dim_magic_operator_cos_{A}_{B}_{C}")
    return high_dim


def high_dim_magic_generator_sin(A: float, B: float, C: float) -> qt.Qobj:
    N = 1000
    print(f"Generating pre-truncation form of the GKP Operator...")
    X, Y, Z, U = get_XYZU_sin(N)
    high_dim = A * X + B * Y + C * Z + np.sqrt(A**2 + B**2 + C**2) * U
    qt.qsave(high_dim, f"cache/operators/high_dim_magic_operator_sin_{A}_{B}_{C}")
    return high_dim

def high_dim_magic_generator_paper(cx: float, cy: float, cz: float) -> qt.Qobj:
    N = 2500
    print(f"Generating pre-truncation form of the GKP Operator...")
    X, Y, Z, U = get_XYZU_paper(N)
    high_dim = cx * X + cy * Y + cz * Z + U
    qt.qsave(high_dim, f"cache/operators/high_dim_magic_operator_paper_{cx}_{cy}_{cz}")
    return high_dim


def magic_operator_sin(
    N: int, A: float = 1.0, B: float = 1.0, C: float = 1.0
) -> qt.Qobj:

    filename = f"cache/operators/high_dim_magic_operator_sin_{A}_{B}_{C}"

    if os.path.isfile(f"{filename}.qu"):
        high_dim = qt.qload(filename)
    else:
        high_dim = high_dim_magic_generator_sin(A, B, C)

    return qt.Qobj(high_dim.full()[:N, :N])


def magic_operator_cos(
    N: int, A: float = 1.0, B: float = 1.0, C: float = 1.0
) -> qt.Qobj:

    filename = f"cache/operators/high_dim_magic_operator_cos_{A}_{B}_{C}"

    if os.path.isfile(f"{filename}.qu"):
        high_dim = qt.qload(filename)
    else:
        high_dim = high_dim_magic_generator_cos(A, B, C)

    return qt.Qobj(high_dim.full()[:N, :N])


def magic_operator_paper(
    N: int, A: float = 1.0, B: float = 1.0, C: float = 1.0
) -> qt.Qobj:

    filename = f"cache/operators/high_dim_magic_operator_paper_{A}_{B}_{C}"

    if os.path.isfile(f"{filename}.qu"):
        high_dim = qt.qload(filename)
    else:
        high_dim = high_dim_magic_generator_paper(A, B, C)

    return qt.Qobj(high_dim.full()[:N, :N])


def magic_operator(
    N: int,
    A: float = 1.0,
    B: float = 1.0,
    C: float = 1.0,
    mode: Literal["sin", "cos", "paper"] = "paper",
) -> qt.Qobj:
    """
    Generate or load a high-dimensional GKP magic operator.

    Parameters:
        N (int): Dimension of the operator.
        A (float): Coefficient for the X operator.
        B (float): Coefficient for the Y operator.
        C (float): Coefficient for the Z operator.
        mode (str): 'sin' or 'cos' to choose the type of operator.

    Returns:
        qutip.Qobj: The high-dimensional GKP magic operator.
    """
    if mode == "sin":
        return magic_operator_sin(N, A, B, C)
    elif mode == "cos":
        return magic_operator_cos(N, A, B, C)
    elif mode == "paper":
        return magic_operator_paper(N, A, B, C)
    else:
        raise ValueError("Mode must be either 'sin' or 'cos'.")


def check_operator_equivalence(
    N: int, A: float, B: float, C: float, tolerance: float = 1e-10
) -> bool:
    """
    Check if sin and cos versions of the magic operator are equivalent for given parameters.

    Returns:
        bool: True if operators are equivalent within tolerance
    """
    sin_op = magic_operator(N, A, B, C, mode="sin")
    cos_op = magic_operator(N, A, B, C, mode="cos")
    diff = (sin_op - cos_op).full()
    equivalent = np.allclose(diff, np.zeros_like(diff), atol=tolerance)
    if not equivalent:
        trace_diff = np.trace(diff)
        norm_diff = np.linalg.norm(diff)
        max_diff = np.max(np.abs(diff))
        print(f"Operators are NOT equivalent for N={N}, A={A}, B={B}, C={C}")
        print(f"Trace difference: {trace_diff}")
        print(f"Frobenius norm of difference: {norm_diff}")
        print(f"Maximum element-wise difference: {max_diff}")
    return equivalent


def find_optimal_p0(N: int, r: float = 0.0) -> qt.Qobj:
    """
    Find the optimal squeezing for the p-eigenstate projector, given a dimension N.

    This uses an exponentially-bracketed binary search over the squeezing parameter r
    to find the largest r such that the probability mass in the first N Fock levels
    is at least a target threshold (default 0.99). We compute probabilities in a
    moderately larger Hilbert space for accuracy, but reuse the vacuum basis vector
    across evaluations for speed.

    Parameters
    ----------
    N : int
        The dimension of the projector.

    r : float, optional
        The starting value for the squeezing. Defaults to 0.

    Returns
    -------
    Qobj
        The optimal p-eigenstate projector of dimension N.
    """

    threshold_probability = 0.99
    dim_factor = (
        6  # compute mass in a slightly larger space for accuracy/speed tradeoff
    )
    work_dim = max(N, dim_factor * N)

    print(f"Constructing optimal p = 0 projection in N = {N} dimensions...")

    basis_work_0 = qt.basis(work_dim, 0)

    def mass_in_first_N(squeeze_r: float) -> float:
        # Compute squeezed vacuum coefficients and return mass in first N levels
        psi = qt.squeeze(work_dim, squeeze_r) * basis_work_0
        coefs = psi.full().flatten()
        # Use absolute square; coefficients may be complex
        return float(np.sum(np.abs(coefs[:N]) ** 2))

    # Ensure monotone direction matches the original logic (moving r downward reduces mass)
    # Find bracket [r_min, r_max] with mass(r_min) < threshold <= mass(r_max)
    r_max = float(r)
    p_max = mass_in_first_N(r_max)

    if p_max < threshold_probability:
        # Increase r exponentially until we exceed threshold
        step = 0.1
        while p_max < threshold_probability:
            r_max += step
            p_max = mass_in_first_N(r_max)
            step *= 1.5

    # Now walk downward to find r_min with mass below threshold
    r_min = r_max
    p_min = p_max
    step = 0.1
    while p_min >= threshold_probability:
        r_min -= step
        p_min = mass_in_first_N(r_min)
        step *= 1.5

    # Binary search within [r_min, r_max]
    tol_r = 1e-3
    max_iter = 50
    for _ in range(max_iter):
        if r_max - r_min <= tol_r:
            break
        r_mid = 0.5 * (r_min + r_max)
        p_mid = mass_in_first_N(r_mid)
        if p_mid >= threshold_probability:
            r_max = r_mid
        else:
            r_min = r_mid

    optimal_r = r_max
    print(f"optimal p eigenket squeezing for N = {N} is {optimal_r:.4f}")

    # Compute the final squeezed state for N and save projector
    projector = qt.ket2dm(qt.squeeze(N, optimal_r) * qt.basis(N, 0))
    qt.qsave(projector, f"cache/operators/p0_projector_N{N}")
    return projector


def p0_projector(N: int, r: float = 0.0) -> qt.Qobj:
    """
    Load or generate the p0 projector for a given dimension N.

    This function checks for a precomputed p0 projector saved in a file. If it
    exists, it loads the projector. Otherwise, it calculates the optimal squeezing
    parameter to generate the projector and saves it for future use.

    Parameters
    ----------
    N : int
        The dimension of the Fock space.
    r : float, optional
        Initial squeezing parameter guess (default is 0.0).

    Returns
    -------
    Qobj
        The truncated p0 projector for the specified dimension N.
    """
    filename = f"cache/operators/p0_projector_N{N}"

    if os.path.isfile(f"{filename}.qu"):
        high_dim = qt.qload(filename)
    else:
        high_dim = find_optimal_p0(N, r)

    return qt.Qobj(high_dim.full()[:N, :N])


Number = Union[float, int]
ComplexLike = Union[complex, Number]


def bloch_from_ab(
    ab: Tuple[Number, ComplexLike], normalize: bool = False, tol: float = 1e-9
) -> Tuple[float, float, float]:
    """
    Convert (a,b) -> (c_x, c_y, c_z) where
      a = cos(theta),
      b = e^{i phi} sin(theta).

    Formula used:
      c_z = 1 - 2*a**2
      c_x + i c_y = -2 * a * b

    Parameters
    ----------
    ab : tuple (a, b)
        a : real number (cos(theta))
        b : complex-like (e^{i phi} sin(theta))
    normalize : bool
        If True, and if a**2 + |b|**2 differs slightly from 1 due to
        numerical error, rescale b so a**2 + |b|**2 == 1.
    tol : float
        Tolerance for checking normalization.

    Returns
    -------
    (c_x, c_y, c_z) as floats
    """
    a, b = ab
    a = float(a)
    b = complex(b)

    if np.abs(a) > 1 + 1e-12:
        raise ValueError(f"a = {a} has |a|>1 (not a valid cos theta)")

    # optional normalization if small numerical error present
    norm_diff = a * a + (np.abs(b) ** 2) - 1.0
    if normalize and np.abs(norm_diff) > tol:
        if np.abs(b) == 0:
            # if b is zero, then sin(theta)=0 and a==±1; nothing to scale
            pass
        else:
            scale = np.sqrt(max(0.0, 1.0 - a * a)) / np.abs(b)
            b *= scale
    elif abs(norm_diff) > 1e-6:
        # warn the user (not raising) if it's noticeably off
        import warnings

        warnings.warn(
            f"a^2 + |b|^2 = {a*a + abs(b)**2:.6g} != 1 (tol={tol}). "
            "Check inputs or enable normalize=True."
        )

    cz = 1.0 - 2.0 * (a * a)
    cx_cy = -2.0 * a * b
    cx = float(cx_cy.real)
    cy = float(cx_cy.imag)

    return (cx, cy, cz)



def catfid(N, state, a, c, k, projector):

    operator = operator_new(N, a, 0, c, k)
    cat_squeezing = qt.expect(operator, state)
    ideal_state = (
        (qt.displace(N, a / 2) + qt.displace(N, -a / 2))
        * qt.squeeze(N, np.log(2) / 2)
        * qt.basis(N, 0)
    ).unit()
    output_state = measure_mode(
        N, beam_splitter(N, state, qt.basis(N, 0)), projector, 1
    )
    output_fidelity = qt.fidelity(output_state, ideal_state)

    return (cat_squeezing, output_fidelity)


def catgkp(N, state, rounds, c, k, projector):

    u = 2 * np.sqrt(2) * np.sqrt(np.pi) * 2 ** ((rounds - 3) / 2)
    operator = operator_new(N, u, 0, c, k)
    cat_squeezing = qt.expect(operator, state)
    gkp_state = breeding(N, rounds, state, projector)
    gkp_operator = gkp_operator_new(N)
    gkp_squeezing = qt.expect(gkp_operator, gkp_state)

    return (cat_squeezing, gkp_squeezing)


def construct_initial_state(N, desc, param):
    # Handle Fock states: either "fock" with param as photon number, or direct integer
    if desc == "fock":
        n = int(param)
        if n < 0 or n >= N:
            raise ValueError(f"Fock state photon number {n} must be >= 0 and < N={N}")
        return qt.basis(N, n), 0.1**n
    
    # Legacy support: try to parse desc as integer (Fock state)
    try:
        n = int(desc)
        if n < 0 or n >= N:
            raise ValueError(f"Fock state photon number {n} must be >= 0 and < N={N}")
        return qt.basis(N, n), 0.1**n
    except (ValueError, TypeError):
        pass
    
    # Handle other predefined states
    if desc == "vacuum":
        return qt.basis(N, 0), 1.0
    elif desc == "coherent":
        print(f"Coherent state with alpha = {param}, type: {type(param)}")
        return qt.coherent(N, param), 1.0
    elif desc == "squeezed":
        return qt.squeeze(N, param) * qt.basis(N, 0), 1.0
    elif desc == "displaced":
        return qt.displace(N, param) * qt.basis(N, 0), 1.0
    elif desc == "cat":
        return (
            (qt.displace(N, param / 2) + qt.displace(N, -param / 2))
            * qt.basis(N, 0)
        ).unit(), 1.0
    elif desc == "squeezed_cat":
        return operator_new(N, param, 0, 10, 100).groundstate()[1], 1.0
    else:
        raise ValueError(f"Unknown initial state description: {desc}. Supported: 'vacuum', 'coherent', 'squeezed', 'displaced', 'cat', 'squeezed_cat', 'fock', or integer (Fock state)")



def construct_operator(
    N: int, target_superposition: Union[qt.Qobj, np.ndarray]
) -> qt.Qobj:
    cx, cy, cz = bloch_from_ab(target_superposition, normalize=True)
    return magic_operator(N, cx, cy, cz)


if __name__ == "__main__":
    N = 30
    target_superposition = (1.0, 1.0)
    operator = construct_operator(N, target_superposition)
    print(operator)