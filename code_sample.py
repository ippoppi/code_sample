"""
This code performs the fit of the gNFW pressure profile to a set of galaxy clusters images.
It uses multi-frequency data from Planck and SPT.
"""

# nohup mpirun -n 5 python -m mpi4py JointFit_gNFW_raympi.py > Logs/JointFit_gNFW_raympi.log &
import numpy as np
from astropy.io import fits
from scipy.signal import convolve2d
from tempfun import *
from scipy.io import readsav
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
from scipy.special import gamma as gammafun
from cobaya.run import run
from cobaya.log import LoggedError
from scipy.signal import fftconvolve
from mpi4py import MPI
import os.path
import time
import ray


def fwhm2sigma(fwhm):
    # Converts beam Full Width Half Maximum to Gaussian sigma
    return fwhm / np.sqrt(8 * np.log(2))


# Radial mean stuff


def avgprof(mappa, binsizeam, nbin, coord):
    # Computes the radial profile of a map by averaging pixels within annular bins
    prof = []
    for i in range(nbin):
        mask = (coord > i * binsizeam) & (coord <= (i + 1.0) * binsizeam)
        prof.append(np.nanmean(mappa[mask]))
    return np.array(prof)


def avgprofrs(mappa, binsizeam, nbin, coord):
    # Same as avgprof, but returns both the profile and the radial centers (rs)
    prof = []
    rs = []
    i = 0
    for i in range(nbin):
        prof.append(
            np.nanmean(
                mappa[(coord > i * binsizeam) & (coord <= (i + 1.0) * binsizeam)]
            )
        )
        rs.append((i + 0.5) * binsizeam)
    return np.array(prof), np.array(rs)


# Likelihood stuff


def constraints(par):
    # Defines constraints for the optimizer.
    theta = np.zeros(5)
    theta[free] = par
    if fixed:
        theta[fix] = parfix
    tpar = np.array(convback(theta))
    return tpar[3] - tpar[4]


def CosmotI(X, M, C, Ci, N=20):
    # Computes the likelihood (from arXiv:1511.05969)
    norm = gammafun(N / 2) / (
        (np.pi * (N - 1)) ** (np.size(M) * 0.5)
        * gammafun((N - np.size(M)) * 0.5)
        * np.sqrt(np.linalg.det(C))
    )
    return norm * (1.0 + np.dot(np.dot((X - M).T, Ci), (X - M)) / (N - 1)) ** (-0.5 * N)


# Read IDL structures
def readidl(filename, key="szcat_ysz", verbose=False):
    # Read IDL .sav files
    data = readsav(file_name=filename, verbose=verbose, python_dict=True)
    if key:
        data = data[key][0]
    return data


# Compute cluster model


def model0(par, r500, z, binsiz, coord, rmax500=8, rmaxz=20):
    # Generates the model profile and projects it to a 2D map template.
    prof, rs = arnaud10_prof(
        r500, par, binsize=binsiz, rmax500=rmax500, rmaxz=rmaxz
    )  # Not A10 anymore!
    rsam = rs * r500toam(1, z, H0=70, Om0=0.3, Ode0=0.7)
    return template(prof, rsam, coord)


@ray.remote
def modelrad(
    par,
    r500,
    z,
    size,
    binrad,
    nbin,
    coord,
    rmax500=8,
    rmaxz=20,
    binsizes=np.array([0.05]),
    window=[0],
    ysz_scal=np.array([1]),
):
    # Core Ray worker function. Executed in parallel on the node.
    # Calculates the model, applies instrumental effects, and bins it.
    s = len(ysz_scal)
    temp = []
    rrr = -1
    for i in range(s):
        if spt[i] != rrr:
            temp0 = model0(par, r500, z, binsizes[i], coord[i], rmax500[i], rmaxz)
        rrr = spt[i]
        # Convolve with beam (Planck)
        if np.size(window[i]) == 1:
            temp.append(
                avgprof(
                    np.sum(
                        atwt2d(ysz_scal[i] * gaussian_filter(temp0, window[i]), 7)[1:],
                        axis=0,
                    ),
                    binrad[i],
                    nbin[i],
                    coord[i],
                )
            )
        else:
            # PSF convolution (SPT)
            temp.append(
                avgprof(
                    ysz_scal[i] * fftconvolve(temp0, window[i], mode="same"),
                    binrad[i],
                    nbin[i],
                    coord[i],
                )
            )
    return P500(
        r500, z, delta=500, m_unit=3.0e14, HO=70, Omega_M=0.3, Lambda0=0.7
    ) * np.array(temp)


def B3splinefast2d(X, step):
    # B3 spline convolution for wavelet transform
    C1 = 1.0 / 16.0
    C2 = 1.0 / 4.0
    C3 = 3.0 / 8.0
    KSize = int(4 * step + 1)
    KS2 = KSize // 2
    Kernel = np.zeros(KSize)
    Kernel[0] = C1
    Kernel[KSize - 1] = C1
    Kernel[KS2 + step] = C2
    Kernel[KS2 - step] = C2
    Kernel[KS2] = C3
    Kernel = Kernel.reshape((len(Kernel), 1))
    out = convolve2d(X, Kernel, mode="same")
    return convolve2d(out, Kernel.T, mode="same")


def atwt2d(X, nscale):
    # Computes the "A Trous" Wavelet Transform (decomposition into scales)
    wt = np.zeros((nscale, len(X), len(X)))
    Xi = np.ones_like(X) * X
    step = 1
    for i in range(nscale - 1):
        Xii = B3splinefast2d(Xi, step)[: len(Xi)]

        wt[-i - 1, :, :] = Xi - Xii
        Xi = Xii
        step *= 2
    wt[0, :, :] = Xii
    return wt


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Initialize Ray with 40 CPUs (assuming 1 process per node with 40 cores)
#!!!! To adapt for different computing cluster !!!!
ray.init(num_cpus=40)
savesuff = "output"
savefile = "path/to/output" + savesuff

# Load the cluster catalog
cat = fits.open("path/to/catalog")[1].data


ncluster = len(cat)
# Initialize global lists to store data for all clusters
tglobs = []
tcovar = []
ticovar = []
models = []
r5s = []
zs = []
dimos = []
binrads = []
nbins = []
coordrads = []
r5maxs = []
binsizess = []
windows = []
yszscals = []
rmaxzs = []

m5s = cat["M500"]

spt_hr = True

# Define which channels to fit (subset of available data)
channel = [0, 1, 3, 4, 6]
fix = [1, 4]  # Parameters to keep fixed
free = [0, 1, 2, 3, 4]  # Parameters to fit

#  Define Instrument Constants
data_dir = "path/to/data"
data_sdi = ["/full/", "/"]
data_suf = ["_HFI_PR2_", "_spt_equat_05am_ptsmask_"]
data_end = ["_eq_05am_512", "_512"]

# Frequencies
st_f = ["95", "150", "220", "100", "143", "217", "353", "545", "857"]
st_f = [st_f[c] for c in channel]  # Filter for selected channels

# SPT boolean mask (1=SPT, 0=Planck)
spt = [1, 1, 1, 0, 0, 0, 0, 0, 0]
spt = [spt[c] for c in channel]

# Units conversion factors
un = [1.0, 1.0, 1.0, 1.0e6, 1.0e6, 1.0e6, 1.0e6, 1.0e6, 1.0e6]
un = [un[c] for c in channel]

# Beam parameters (FWHM -> Sigma)
beamf = [9.69, 7.30, 5.02, 5.02, 5.02, 4.64]
beam = [fwhm2sigma(b) for b in beamf]
nspt = np.sum(np.array(spt)).astype("int")
beam = [beam[c - 3] for c in channel[nspt:]]

beamf = [1.75 for i in range(nspt)] + [beamf[c - 3] for c in channel[nspt:]]

# Ysz Scaling factors (Temperature to y-parameter conversion)
ysz_scal = np.array(
    [
        -1.513641616933553 * 2.725e6,
        -0.9019438947233195 * 2.725e6,
        0.045404401335390615 * 2.725e6,
        -4.0304499,
        -2.7822697,
        0.19408140,
        6.2064362,
        839.13898,
        59.771610,
    ]
)
ysz_scal = [ysz_scal[c] for c in channel]

# Map resolution settings
if spt_hr:
    resol = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    dimorg = np.ones_like(resol) * 256
else:
    resol = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    dimorg = np.ones_like(resol) * 256
resol = [resol[c] for c in channel]

# Mass integration radius settings
if spt_hr:
    rmass = [4, 4, 4, 6, 6, 6, 6, 6, 6]
else:
    rmass = [4, 4, 4, 8, 6, 6, 6, 6, 6]
rmass = np.array([rmass[e] for e in channel])

# Prepare fixed/free parameter lists
for f in fix:
    free.remove(f)
fixed = len(fix) > 0

# --- Start Cluster Loop ---
ntru = 0
for ntarget in range(0, ncluster):
    ntru += 1

    cluster = cat[ntarget]["SPT_ID"]

    # Locate the IDL .sav file for this cluster
    prof_name = (
        "path/to/data"
        + cluster.lower()
        + "....sav"
    )
    if not os.path.isfile(prof_name):
        print("No file! skip")
        continue

    # Load cluster data
    dat = readidl(prof_name, key="szcat_ysz", verbose=False)
    z = dat.redshift
    r500 = dat.r500
    r500am = r500toam(r500, z)

    if rank == 0:
        print(ntarget, ntru, cluster)

    # --- Load Maps & Noise ---
    fmaps = []
    psf = []
    sptnois = []
    Cmbrec = []

    for i in range(len(st_f)):
        # Construct file path
        file_map = (
            data_dir
            + cluster.lower()
            + data_sdi[spt[i]]
            + cluster
            + data_suf[spt[i]]
            + st_f[i]
            + data_end[spt[i]]
            + ".fits"
        )

        if spt[i]:
            # === SPT Data Loading ===
            if spt_hr:
                # High-res file paths
                file_noise = (
                    data_dir
                    + cluster.lower()
                    + "/"
                    + cluster
                    + "_spt_noise_equat_05am_ptsmask_"
                    + st_f[i]
                    + "_512.fits"
                )
                file_psf = (
                    data_dir
                    + cluster.lower()
                    + "/"
                    + cluster
                    + "_spt_tf_eq_05am_"
                    + st_f[i]
                    + "GHz_512.fits"
                )
                file_cmb = (
                    data_dir
                    + cluster.lower()
                    + data_sdi[spt[i]]
                    + cluster
                    + data_suf[spt[i]]
                    + "CMB_noT_MioP13_ZSPT_"
                    + st_f[i]
                    + data_end[spt[i]]
                    + ".fits"
                )
            else:
                # Standard-res file paths
                file_noise = (
                    data_dir
                    + cluster.lower()
                    + "/"
                    + cluster
                    + "_spt_noise_equat_1am_ptsmask_"
                    + st_f[i]
                    + "_512.fits"
                )
                file_psf = (
                    data_dir
                    + cluster.lower()
                    + "/"
                    + cluster
                    + "_spt_tf_n_eq_1am_"
                    + st_f[i]
                    + "GHz_256.fits"
                )

            # Read FITS files for SPT
            sptnois.append(fits.open(file_noise)[0].data[:, 128:-128, 128:-128])
            mapf = fits.open(file_map)[0].data[:, 128:-128, 128:-128]
            Cmbrec.append(fits.open(file_cmb)[0].data)

            # Load SPT PSF
            if spt_hr:
                psft = fits.open(file_psf)[0].data[128:-128, 128:-128]
            else:
                psft = fits.open(file_psf)[0].data
            psf.append(psft)
        else:
            # Planck Data Loading
            if spt_hr:
                mapf = fits.open(file_map)[0].data[:, 128:-128, 128:-128]
            else:
                mapf = fits.open(file_map)[0].data

            Cmbrec = None
            sptnois = None

        fmaps.append(mapf[0])

    # --- Geometry & Binning Calculation (Cluster Dependent) ---
    if spt_hr:
        nbinspt = 11
        r500binspt = 0.3
    else:
        nbinspt = 11
        r500binspt = 0.35
    nbinplanck = 8
    r500binplanck = 0.7

    # Calculate radial bin sizes in arcmin based on r500am
    binsizesam = []
    for i in range(len(channel)):
        if spt[i]:
            bsam = r500am * r500binspt
            if bsam >= 0.5:
                binsizesam.append(bsam)
            else:
                binsizesam.append(0.5)
                nbinspt = int(r500am * 4 / 0.5 + 0.5)
        else:
            binsizesam.append(r500am * r500binplanck)

    nbin = []
    for i in range(len(channel)):
        if spt[i]:
            nbin.append(nbinspt)
        else:
            nbin.append(nbinplanck)

    # Calculate map dimensions
    dimo = np.array(
        [
            np.max((np.int(30 / r), np.int(np.round(2 * r500am * rm / r))))
            for r, rm in zip(resol, rmass)
        ]
    )

    for i in range(len(dimo)):
        if np.mod(dimo[i], 2) != 0:
            dimo[i] += 1

    binsizes = np.array(resol) / r500toam(1, z)

    # Compute Data Profiles & Covariance
    covar = []
    icovar = []
    glcoord = []
    planck_dir = "path/to/data"
    globs = []
    npsf = []
    radi = []

    for i in range(len(channel)):
        size = dimo[i]
        if spt[i]:
            # Process SPT channel
            coord = make_coord_map(resol[i], size)
            glcoord.append(coord)
            cut = np.int((dimorg[i] - size) / 2)
            mask = np.where(
                fmaps[i][cut:-cut, cut:-cut] == 0, np.nan, 1.0
            )  # Uses index 'i'

            # Compute radial profile of data
            ma, rs = avgprofrs(
                (fmaps[i][cut:-cut, cut:-cut] - Cmbrec[i][cut:-cut, cut:-cut]) * mask,
                binsizesam[i],
                nbin[i],
                coord,
            )
            globs.append(ma)

            # Calculate covariance from noise simulations
            ecc = []
            for j in range(20):
                ppp = avgprof(
                    sptnois[i][j, cut:-cut, cut:-cut] * mask,
                    binsizesam[i],
                    nbin[i],
                    coord,
                )
                ecc.append(ppp)
            covar.append(np.cov(np.array(ecc).T))
            icovar.append(np.linalg.inv(np.cov(np.array(ecc).T)))
            # Store 2D PSF
            npsf.append((psf[i][cut:-cut, cut:-cut]).astype("float64"))
        else:
            # Process Planck channel
            cut = np.int((256 - size) / 2)
            coord = make_coord_map(resol[i], size)
            glcoord.append(coord)

            # Load specific Planck files
            file_map = (
                planck_dir + cluster + "_" + st_f[i] + "GHz_SPTxPlanck_MioP13_ZSPT.fits"
            )
            file_noise = (
                planck_dir
                + cluster
                + "_"
                + st_f[i]
                + "GHz_SPTxPlanck_noises_MioP13_ZSPT.fits"
            )
            mapf = fits.open(file_map)[0].data
            mask = np.where(mapf[0, cut:-cut, cut:-cut] == 0, np.nan, 1.0)
            nosf = fits.open(file_noise)[0].data

            # Compute radial profile
            ma, rs = avgprofrs(
                mapf[0, cut:-cut, cut:-cut] * mask, binsizesam[i], nbin[i], coord
            )
            globs.append(ma)

            # Calculate covariance
            ecc = []
            for j in range(100):
                ppp = avgprof(
                    nosf[j, cut:-cut, cut:-cut] * mask, binsizesam[i], nbin[i], coord
                )
                ecc.append(ppp)
            covar.append(np.cov(np.array(ecc).T))
            icovar.append(np.linalg.inv(np.cov(np.array(ecc).T)))

        radi.append(rs)

    # Store beam for Planck channels (scalar sigma)
    npsf += beam
    psf = npsf

    # Accumulate cluster data into global lists
    tglobs.append(np.array(globs))
    tcovar.append(covar)
    ticovar.append(icovar)
    r5s.append(r500)
    zs.append(z)
    dimos.append(dimo)
    binrads.append(binsizesam)
    nbins.append(nbin)
    coordrads.append(glcoord)
    r5maxs.append(np.array(rmass) * np.sqrt(2))
    rmaxzs.append(20)
    binsizess.append(binsizes)
    windows.append(psf)
    yszscals.append(ysz_scal)

# --- FITTING & SAMPLING ---

# Parameter conversion logic (Log scaling, inversions)
ncluster = ntru
parro = [6.41, 1.81, 1.33, 4.13, 0.31]  # Starting values (P13)
convpar = lambda x: [np.log10(x[0]), x[1], 1.0 / x[2], x[3], x[4]]
convback = lambda x: [10 ** x[0], x[1], 1.0 / x[2], x[3], x[4]]
paro = convpar(parro)
parml = [paro[f] for f in free]
parfix = [paro[j] for j in fix]


def modelli(par):
    # Distributed model computation using Ray
    a = [
        modelrad.remote(
            par,
            r5s[i],
            zs[i],
            dimos[i],
            binrads[i],
            nbins[i],
            coordrads[i],
            r5maxs[i],
            rmaxzs[i],
            binsizess[i],
            windows[i],
            yszscals[i],
        )
        for i in range(ncluster)
    ]
    return ray.get(a)  # Gather results


# Set parameter priors/limits
lowso = [10 ** (-2.0), 0.2, 4 * 10 ** (-1.0), 0.0, 0.0]
higho = [100, 5.0, 10.0, 15.0, 10]
lows = np.array(convpar(lowso))
highs = np.array(convpar(higho))

bndsml = [(np.min((h, l)), np.max((h, l))) for l, h in zip(lows, highs)]
bndsml = [bndsml[f] for f in free]


def likc(par):
    # Combined Log-Likelihood function
    theta = np.zeros(5)
    theta[free] = par
    if fixed:
        theta[fix] = parfix
    tpar = np.array(convback(theta))

    # \beta > \gamma condition
    if tpar[4] > tpar[3]:
        return -np.inf

    # Compute models for all clusters in parallel
    modes = modelli(tpar)
    fulik = 0

    # Sum log-likelihoods over all clusters and channels
    for ii in range(ncluster):
        lik = 0.0
        for il in range(len(modes[ii])):
            if spt[il]:
                # SPT Likelihood
                lik += np.log(
                    CosmotI(
                        tglobs[ii][il], modes[ii][il], tcovar[ii][il], ticovar[ii][il]
                    )
                )
            else:
                # Planck Likelihood
                lik += np.log(
                    CosmotI(
                        tglobs[ii][il],
                        modes[ii][il],
                        tcovar[ii][il],
                        ticovar[ii][il],
                        N=100,
                    )
                )
        fulik += lik

    return fulik


nll = lambda p: -likc(p)

# Minimizer Setup
ftol = 2.220446049250313e-09
con = lambda *args: constraints(*args)

if rank == 0:
    start_time = time.time()
    print("")
    # Run Minimization to find best starting point
    res = minimize(
        nll, parml, bounds=bndsml, constraints={"type": "ineq", "fun": con}, tol=ftol
    )
    print("--- %s seconds ---" % (time.time() - start_time))
    print(res["x"])
else:
    res = None

# Broadcast result to all MPI nodes
res = comm.bcast(res, root=0)

likeradC = lambda logp, ainv, beta: likc([logp, ainv, beta])


# Cobaya MCMC Configuration
cobinput = {
    "likelihood": {"Joint_lik": likeradC},
    "params": {
        "logp": {
            "prior": {"min": lows[0], "max": highs[0]},
            "ref": {"dist": "norm", "loc": res["x"][0], "scale": 0.005},
            "latex": r"Log(P_0)",
        },
        "ainv": {
            "prior": {"min": highs[2], "max": lows[2]},
            "ref": {"dist": "norm", "loc": res["x"][1], "scale": 0.01},
            "latex": r"\alpha^{-1}",
        },
        "beta": {
            "prior": {"min": lows[3], "max": highs[3]},
            "ref": {"dist": "norm", "loc": res["x"][2], "scale": 0.01},
            "latex": r"\beta",
        },
        "p": {"derived": lambda logp: 10**logp, "latex": r"P_{0}"},
        "alpha": {"derived": lambda ainv: 1.0 / ainv, "latex": r"\alpha"},
    },
    "sampler": {
        "mcmc": {
            "max_tries": 1000,
            "Rminus1_stop": 0.01,
            "Rminus1_cl_stop": 0.1,
            "burn_in": 30,
            "learn_proposal": True,
        }
    },
    "output": savefile,
    "force": True,
    "resume": False,
}

# Run Cobaya Sampling
success = False
try:
    upd_info, mcmc = run(cobinput)
    success = True
except LoggedError as err:
    pass

# Did it work?
success = all(comm.allgather(success))

if not success and rank == 0:
    print("Sampling failed!")
