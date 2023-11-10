import numpy as np
from tqdm import tqdm


def simulate_gxe(
    scenario: int,
    n_rep: int = 100,
    n_indiv: int = 10000,
    n_gwas: int = 20000,
    n_snp=10000,
    hsq: float = None,
    rg: float = None,
    hsq1: float = None,
    hsq2: float = None,
    prop_amp: float = None,
):
    """Simulate three modes of GxE as in Durvasula & Price 2023

    Parameters
    ----------
    scenario : int
        which scenario, 1,2,3
    n_rep : int, optional
        number of replicates, by default 100
    n_indiv : int, optional
        number of individuals, by default 10000
    n_gwas : int, optional
        number of GWAS individuals, used to simulate GWAS noise, by default 20000
    n_snp : int, optional
        number of SNPs to simulate, by default 10000
    hsq : float, optional
        heritability, by default None
    rg : float, optional
        genetic correlation, by default None
    hsq1 : float, optional
        heritability in pop1, by default None
    hsq2 : float, optional
        heritability in pop2, by default None
    prop_amp : float, optional
        proportional amplification magnitude, by default None

    Returns
    -------
    ymat (n_indiv, 2), prsmat, gmat, emat, beta, betaprs
    """
    if scenario == 1:
        assert (hsq is not None) and (rg is not None)
        beta_cov = np.array([[hsq, rg * hsq], [rg * hsq, hsq]]) / n_snp
        e_var = [1 - hsq, 1 - hsq]
    elif scenario == 2:
        assert (hsq1 is not None) and (hsq2 is not None)
        beta_cov = (
            np.array([[hsq1, np.sqrt(hsq1 * hsq2)], [np.sqrt(hsq1 * hsq2), hsq2]])
            / n_snp
        )
        e_var = [1 - hsq1, 1 - hsq2]
    elif scenario == 3:
        assert (hsq is not None) and (prop_amp is not None)
        beta_cov = np.array([[hsq, hsq], [hsq, hsq]]) / n_snp
        e_var = [1 - hsq, 1 - hsq]
    else:
        raise NotImplementedError

    # generate standardized genotypes
    xmat1 = np.random.binomial(n=2, p=0.5, size=(n_indiv, n_snp))
    xmat2 = np.random.binomial(n=2, p=0.5, size=(n_indiv, n_snp))
    xmat1 = (xmat1 - xmat1.mean(axis=0)) / xmat1.std(axis=0)
    xmat2 = (xmat2 - xmat2.mean(axis=0)) / xmat2.std(axis=0)

    # simulate effect sizes
    if hsq1 is None:
        hsq1 = hsq
    beta = np.zeros((n_snp, 2, n_rep))
    betaprs = np.zeros((n_snp, n_rep))
    emat = np.zeros([n_indiv, 2, n_rep])

    for i in tqdm(range(n_rep)):
        beta_i = np.random.multivariate_normal(mean=[0, 0], cov=beta_cov, size=n_indiv)

        # generate betahat and betaprs using pop1
        betahat_i = beta_i[:, 0] + np.random.normal(
            0, scale=np.sqrt((1 - hsq1 / n_snp) / n_gwas), size=n_snp
        )
        betaprs_i = betahat_i * hsq1 / (hsq1 + n_snp / n_gwas)
        beta[:, :, i] = beta_i
        betaprs[:, i] = betaprs_i
        emat[:, :, i] = np.random.normal(scale=np.sqrt(e_var), size=(n_indiv, 2))

    gmat = np.zeros([n_indiv, 2, n_rep])
    prsmat = np.zeros([n_indiv, 2, n_rep])

    gmat[:, 0, :] = xmat1 @ beta[:, 0, :]
    gmat[:, 1, :] = xmat2 @ beta[:, 1, :]
    prsmat[:, 0, :] = xmat1 @ betaprs
    prsmat[:, 1, :] = xmat2 @ betaprs
    ymat = gmat + emat

    if scenario == 3:
        assert prop_amp is not None
        ymat[:, 1, :] *= prop_amp
    return ymat, prsmat, gmat, emat, beta, betaprs


def simulate_gxe2(
    scenario,
    n_rep=100,
    n_indiv=10000,
    hsq=None,
    rg=None,
    hsq1=None,
    hsq2=None,
    prop_amp=None,
):
    """simulate GxE (directly for individuals)"""
    if scenario == 1:
        assert (hsq is not None) and (rg is not None)
        g_cov = np.array([[hsq, rg * hsq], [rg * hsq, hsq]])
        e_var = [1 - hsq, 1 - hsq]
    elif scenario == 2:
        assert (hsq1 is not None) and (hsq2 is not None)
        g_cov = np.array([[hsq1, np.sqrt(hsq1 * hsq2)], [np.sqrt(hsq1 * hsq2), hsq2]])
        e_var = [1 - hsq1, 1 - hsq2]
    elif scenario == 3:
        assert (hsq is not None) and (prop_amp is not None)
        g_cov = np.array([[hsq, hsq], [hsq, hsq]])
        e_var = [1 - hsq, 1 - hsq]
    else:
        raise NotImplementedError

    gmat = np.zeros([n_indiv, 2, n_rep])
    emat = np.zeros([n_indiv, 2, n_rep])

    for i in range(n_rep):
        g = np.random.multivariate_normal(mean=[0, 0], cov=g_cov, size=n_indiv)
        gmat[:, :, i] = g
        e = np.random.normal(scale=np.sqrt(e_var), size=(n_indiv, 2))
        emat[:, :, i] = e
    ymat = gmat + emat

    if scenario == 3:
        ymat[:, 1, :] *= prop_amp
    return ymat, gmat, emat
