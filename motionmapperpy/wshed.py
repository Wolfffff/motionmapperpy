import glob
import time

import h5py
import hdf5storage
import time
from skimage.segmentation import watershed
from skimage.filters import roberts
from matplotlib import pyplot as plt

from sklearn.mixture import GaussianMixture

from .mmutils import *
from datetime import datetime

bmapcmap = gencmap()


# TODO: Rewrite this to use just training data -- pass bounds
def wshedTransform(
    zValues,
    min_regions,
    sigma,
    tsnefolder,
    saveplot=True,
    training=False,
    use_awkde=False,
    alpha=None,
    glob_bw=None,
):
    date = datetime.now().strftime("%Y%m%d")
    print("Starting watershed transform...")

    if use_awkde:
        bounds, xx, density = findPointDensity_awkde(
            zValues,
            alpha,
            glob_bw,
            610,
            rangeVals=[-np.abs(zValues).max() - 15, np.abs(zValues).max() + 15],
        )
        density[density < 8e-6] = 0
    else:
        bounds, xx, density = findPointDensity(
            zValues,
            sigma,
            610,
            rangeVals=[-np.abs(zValues).max() - 15, np.abs(zValues).max() + 15],
        )

    wshed = watershed(-density, connectivity=10)
    wshed[density < 8e-6] = 0

    numRegs = len(np.unique(wshed)) - 1

    # if numRegs < min_regions - 10:
    #     raise ValueError(
    #         "\t Starting sigma %0.1f too high, maximum # wshed regions possible is %i."
    #         % (sigma, numRegs)
    #     )

    while numRegs > min_regions and not use_awkde:
        sigma += 0.05
        _, xx, density = findPointDensity(
            zValues,
            sigma,
            611,
            rangeVals=[-np.abs(zValues).max() - 15, np.abs(zValues).max() + 15],
        )
        if training:
            density[density < 8e-6] = 0
            wshed = watershed(-density, connectivity=10)
            wshed[density < 8e-6] = 0
        else:
            wshed = watershed(-density, connectivity=10)

        numRegs = len(np.unique(wshed)) - 1
        print("\t Sigma %0.2f, Regions %i" % (sigma, numRegs), end="\r")
    for i, wreg in enumerate(np.unique(wshed)):
        wshed[wshed == wreg] = i
    wbounds = np.where(roberts(wshed).astype("bool"))
    wbounds = (wbounds[1], wbounds[0])
    if saveplot:
        bend = plt.get_backend()
        plt.switch_backend("Agg")

        fig, axes = plt.subplots(1, 2, figsize=(10, 6))
        fig.subplots_adjust(0, 0, 1, 1, 0, 0)
        ax = axes[0]
        ax.imshow(randomizewshed(wshed), origin="lower", cmap=bmapcmap)
        for i in np.unique(wshed)[1:]:
            fontsize = 8
            xinds, yinds = np.where(wshed == i)
            ax.text(
                np.mean(yinds) - fontsize,
                np.mean(xinds) - fontsize,
                str(i),
                fontsize=fontsize,
                fontweight="bold",
            )
        ax.axis("off")

        ax = axes[1]
        ax.imshow(density, origin="lower", cmap=bmapcmap)
        ax.scatter(wbounds[0], wbounds[1], color="k", s=0.1)
        ax.axis("off")
        fig.savefig(
            f"{tsnefolder}{date}_sigma{str(round(sigma,3)).replace('.','_')}_regions{numRegs}_zWshed.png"
        )
        plt.close()
        plt.switch_backend(bend)
    return wshed, wbounds, sigma, xx, density


def velGMM(ampV, parameters, projectPath, saveplot=True, minimum_regions=50):
    date = datetime.now().strftime("%Y%m%d")
    if parameters.method == "TSNE":
        if parameters.waveletDecomp:
            tsnefolder = projectPath + "/TSNE/"
        else:
            tsnefolder = projectPath + "/TSNE_Projections/"
    else:
        tsnefolder = projectPath + "/UMAP/"
    ampVels = ampV * parameters["samplingFreq"]
    vellog10all = np.log10(ampVels[ampVels > 0])
    npoints = min(50000, len(vellog10all))

    vellog10 = np.random.choice(vellog10all, size=npoints, replace=False)

    gm = GaussianMixture(
        n_components=2, verbose=1, tol=1e-5, max_iter=2000, n_init=1, reg_covar=1e-3
    )
    inds = np.random.randint(0, vellog10.shape[0], size=npoints)
    gm = gm.fit(vellog10[inds, None])
    minind = np.argmin(gm.means_.squeeze())

    if saveplot:
        bend = plt.get_backend()
        plt.switch_backend("Agg")
        fig, ax = plt.subplots(figsize=(8, 8))
        bins = ax.hist(vellog10, bins=200, density=True, color="k", alpha=0.5)
        bins = bins[1]
        p_score = np.exp(gm.score_samples(bins[:, None]))
        ax.plot(bins, p_score, color="k", alpha=0.5)

        for c, compno, mu, sigma, p in zip(
            ["royalblue", "firebrick"],
            [1, 2],
            gm.means_.squeeze(),
            np.sqrt(gm.covariances_.squeeze()),
            gm.weights_,
        ):
            ax.plot(
                bins,
                getPDF(bins, mu, sigma, p),
                label="Component %i" % compno,
                color=c,
                alpha=0.5,
            )

        ax.plot(bins, gm.predict_proba(bins[:, None])[:, minind], label="pRest")
        ax.axvline(
            bins[
                np.where(
                    gm.predict_proba(bins[:, None])[:, minind]
                    < np.min(parameters.pThreshold)
                )[0][0]
            ],
            color="firebrick",
            label="pRest=%0.2f" % np.min(parameters.pThreshold),
        )
        ax.legend()
        ax.set_xlabel(r"$log_{10}$ Velocity")
        ax.set_ylabel("PDF")
        fig.savefig(
            f"{tsnefolder}{date}_sigma{str(round(sigma,3)).replace('.','_')}_minregions{minimum_regions}_zVelocity.png"
        )
        plt.close()
        plt.switch_backend(bend)

    pRest = np.zeros_like(ampVels)
    pRest[ampVels == 0] = 0.0
    pRest[ampVels > 0] = gm.predict_proba(vellog10all[:, None])[:, minind]
    return ampV, pRest


def makeGroupsAndSegments(watershedRegions, zValLens, min_length=10, max_length=100):
    inds = np.zeros_like(watershedRegions)
    start = 0
    for l in zValLens:
        inds[start : start + l] = np.arange(l)
        start += l
    vinds = np.digitize(
        np.arange(watershedRegions.shape[0]),
        bins=np.concatenate([[0], np.cumsum(zValLens)]),
    )

    splitinds = np.where(np.diff(watershedRegions, axis=0) != 0)[0] + 1
    inds = [
        i
        for i in np.split(inds, splitinds)
        if len(i) > min_length and len(i) < max_length
    ]
    wregs = [
        i[0]
        for i in np.split(watershedRegions, splitinds)
        if len(i) > min_length and len(i) < max_length
    ]

    vinds = [
        i
        for i in np.split(vinds, splitinds)
        if len(i) > min_length and len(i) < max_length
    ]
    groups = [np.empty((0, 3), dtype=int)] * watershedRegions.max()

    for wreg, tind, vind in zip(wregs, inds, vinds):
        if wreg == 0:
            continue
        if np.all(vind == vind[0]):
            groups[wreg - 1] = np.concatenate(
                [
                    groups[wreg - 1],
                    np.array([vind[0], tind[0] + 1, tind[-1] + 1])[None, :],
                ]
            )
    groups = np.array([[g] for g in groups])
    return groups


def findWatershedRegions(
    parameters,
    minimum_regions=150,
    startsigma=0.1,
    pThreshold=None,
    saveplot=True,
    endident="*_pcaModes.mat",
    min_length_videos=10,
    prev_wshed_file=None,
):
    date = datetime.now().strftime("%Y%m%d")
    projectionfolder = parameters.projectPath + "/Projections/"
    if parameters.method == "TSNE":
        if parameters.waveletDecomp:
            tsnefolder = parameters.projectPath + "/TSNE/"
        else:
            tsnefolder = parameters.projectPath + "/TSNE_Projections/"
    elif parameters.method == "UMAP":
        tsnefolder = parameters.projectPath + "/UMAP/"
    else:
        raise ValueError("parameters.method can only take values 'TSNE' or 'UMAP'")

    if pThreshold is None:
        parameters.pThreshold = [0.33, 0.67]
    else:
        parameters.pThreshold = pThreshold

    zValues = []
    projfiles = glob.glob(projectionfolder + "/*" + endident)
    t1 = time.time()

    zValNames = []
    zValLens = []
    ampVels = []
    for pi, projfile in enumerate(projfiles):
        fname = projfile.split("/")[-1].split(".")[0]
        print(f"Processing {projfile}")
        zValNames.append(fname)
        print(
            "%i/%i Loading embedding for %s %0.02f seconds."
            % (pi + 1, len(projfiles), fname, time.time() - t1)
        )
        if parameters.method == "TSNE":
            zValident = "zVals" if parameters.waveletDecomp else "zValsProjs"
        else:
            zValident = "uVals"
        with h5py.File(projectionfolder + fname + "_%s.mat" % zValident, "r") as h5file:
            shape = h5file["zValues"][:].T.shape
            print(f"shape of zVals: {shape}")
            print(h5file["zValues"][:].T[0:5, :])
            zValues.append(h5file["zValues"][:].T)
        ampVels.append(
            np.concatenate(
                ([0], np.linalg.norm(np.diff(zValues[-1], axis=0), axis=1)), axis=0
            )
        )
        # with h5py.File(projectionfolder + fname + '_zAmps_vel.mat', 'r') as h5file:
        #     ampVels.append(h5file['ampvel'][:].T.squeeze())

        assert zValues[-1].shape[0] == ampVels[-1].shape[0]
        zValLens.append(zValues[-1].shape[0])

    zValues = np.concatenate(zValues, 0)
    ampVels = np.concatenate(ampVels, 0)
    # print(zValLens)
    zValLens = np.array(zValLens)
    # print(zValNames)
    zValNames = np.array(zValNames, dtype=object)
    print(f"zValues shape going into watershed: {zValues.shape}")
    # raise Exception("stop here")
    if prev_wshed_file is not None:
        f = h5py.File(prev_wshed_file, "r")
        xx = f["xx"][:]
        # sigma = f["sigma"][:]
        wbounds_dataset = f["wbounds"][:]

        # Extract the object references
        # object_refs = wbounds_dataset[:]

        # # Unpack the object references
        # x_ref = object_refs[0, 0]
        # y_ref = object_refs[1, 0]

        # # Access the actual data
        # x_data = f[x_ref][:]
        # y_data = f[y_ref][:]
        x_data = wbounds_dataset[0]
        y_data = wbounds_dataset[1]
        wbounds = (x_data, y_data)
        # density = f["density"][:]
        LL = f["LL"][:]
        print(f"zValues shape: {zValues.shape}")
        zValues_for_density = zValues[~np.isnan(zValues).any(axis=1), :]
        print(f"zValues_for_density shape: {zValues_for_density.shape}")

        _, _, _, _, density = wshedTransform(
            zValues_for_density, minimum_regions, startsigma, tsnefolder, saveplot=False
        )
    else:
        LL, wbounds, sigma, xx, density = wshedTransform(
            zValues, minimum_regions, startsigma, tsnefolder, saveplot=False
        )

    if prev_wshed_file is not None:
        wshed = LL.T
        bend = plt.get_backend()
        plt.switch_backend("Agg")

        fig, axes = plt.subplots(1, 2, figsize=(10, 6))
        fig.subplots_adjust(0, 0, 1, 1, 0, 0)
        ax = axes[0]
        ax.imshow(randomizewshed(wshed), origin="lower", cmap=bmapcmap)
        for i in np.unique(wshed)[1:]:
            fontsize = 8
            xinds, yinds = np.where(wshed == i)
            ax.text(
                np.mean(yinds) - fontsize,
                np.mean(xinds) - fontsize,
                str(i),
                fontsize=fontsize,
                fontweight="bold",
            )
        ax.axis("off")

        ax = axes[1]
        ax.imshow(density, origin="lower", cmap=bmapcmap)
        # ax.scatter(wbounds[0], wbounds[1], color="k", s=0.1)
        ax.axis("off")
        fig.savefig(f"{tsnefolder}{date}_tmp_uWshed.png")
        plt.close()
        plt.switch_backend(bend)

        fig, axes = plt.subplots(1, 2, figsize=(10, 6))
        fig.subplots_adjust(0, 0, 1, 1, 0, 0)
        ax = axes[0]
        ax.imshow(randomizewshed(wshed), origin="lower", cmap=bmapcmap)
        for i in np.unique(wshed)[1:]:
            fontsize = 8
            xinds, yinds = np.where(wshed == i)
            ax.text(
                np.mean(yinds) - fontsize,
                np.mean(xinds) - fontsize,
                str(i),
                fontsize=fontsize,
                fontweight="bold",
            )
        ax.axis("off")

        ax = axes[1]
        ax.imshow(density, origin="lower", cmap=bmapcmap)
        ax.scatter(wbounds[0], wbounds[1], color="k", s=0.1)
        ax.axis("off")
        fig.savefig(f"{tsnefolder}{date}_tmp_uWshed_wbounds.png")
        plt.close()
        plt.switch_backend(bend)

    print("Assigning watershed regions...")
    print(f"zValues shape: {zValues.shape}")
    print(f"xx shape: {xx.shape}")
    # A large number that's guaranteed to be bigger than any element in zValues
    watershedRegions = np.full(zValues.shape[0], np.nan)

    # Create a mask of non-NaN elements in zValues
    mask_non_nan = ~np.isnan(zValues).any(axis=1)

    # Compute the digitization only for the non-NaN elements in zValues
    wr_tmp = np.digitize(zValues[mask_non_nan, :], xx.flatten()[:-1])
    print(f"shape of wr_tmp: {wr_tmp.shape}")
    print(f"max watershed region: {watershedRegions.max()}")
    wr_tmp = LL.T[wr_tmp[:, 1], wr_tmp[:, 0]]
    watershedRegions[mask_non_nan] = wr_tmp
    try:
        sigma
    except NameError:
        sigma = np.nan

    if parameters.method == "TSNE":
        print("Calculating velocity distributions...")
        ampVels, pRest = velGMM(
            ampVels,
            parameters,
            parameters.projectPath,
            saveplot=saveplot,
            minimum_regions=minimum_regions,
        )

        outdict = {
            "zValues": zValues,
            "zValNames": zValNames,
            "zValLens": zValLens,
            "sigma": sigma,
            "xx": xx,
            "density": density,
            "LL": LL,
            "watershedRegions": watershedRegions,
            "v": ampVels,
            "pRest": pRest,
            "wbounds": wbounds,
        }
        hdf5storage.write(
            data=outdict,
            path="/",
            truncate_existing=True,
            filename=f"{tsnefolder}{date}_sigma{str(round(sigma,3)).replace('.','_')}_minregions{minimum_regions}_zVals_wShed_groups_usingprev.mat",
            store_python_metadata=False,
            matlab_compatible=True,
        )

        print("\t tempsave done.")

        t1 = time.time()
        print("Adjusting non-stereotypic regions to 0...")
        bwconn = np.convolve(
            (np.diff(watershedRegions) == 0).astype(bool), np.array([True, True])
        )
        pGoodRest = pRest > np.min(parameters.pThreshold)
        badinds = ~np.bitwise_and(bwconn, pGoodRest)
        watershedRegions[badinds] = 0
        print("\t Done. %0.02f seconds" % (time.time() - t1))
    else:
        pRest = 1.0
    outdict = {
        "zValues": zValues,
        "zValNames": zValNames,
        "zValLens": zValLens,
        "sigma": sigma,
        "xx": xx,
        "density": density,
        "LL": LL,
        "watershedRegions": watershedRegions,
        "v": ampVels,
        "pRest": pRest,
        "wbounds": wbounds,
    }
    hdf5storage.write(
        data=outdict,
        path="/",
        truncate_existing=True,
        filename=f"{tsnefolder}{date}_minregions{minimum_regions}_zVals_wShed_groups_prevregions.mat",
        store_python_metadata=False,
        matlab_compatible=True,
    )
    print("\t tempsave done.")
    raise Exception("Not making groups for now -- final save is done.")
    groups = makeGroupsAndSegments(
        watershedRegions, zValLens, min_length=min_length_videos
    )
    outdict = {
        "zValues": zValues,
        "zValNames": zValNames,
        "zValLens": zValLens,
        "sigma": sigma,
        "xx": xx,
        "density": density,
        "LL": LL,
        "watershedRegions": watershedRegions,
        "v": ampVels,
        "pRest": pRest,
        "wbounds": wbounds,
        "groups": groups,
    }
    hdf5storage.write(
        data=outdict,
        path="/",
        truncate_existing=True,
        filename=f"{tsnefolder}{date}_minregions{minimum_regions}_zVals_wShed_groups_prevwshed_finalsave.mat",
        store_python_metadata=False,
        matlab_compatible=True,
    )

    print("All data saved.")


def findWatershedRegions_training(
    parameters,
    minimum_regions=150,
    startsigma=0.1,
    pThreshold=None,
    saveplot=True,
    endident="*_pcaModes.mat",
):
    date = datetime.now().strftime("%Y%m%d")
    projectionfolder = parameters.projectPath + "/Projections/"
    if parameters.method == "TSNE":
        if parameters.waveletDecomp:
            tsnefolder = parameters.projectPath + "/TSNE/"
        else:
            tsnefolder = parameters.projectPath + "/TSNE_Projections/"
    elif parameters.method == "UMAP":
        tsnefolder = parameters.projectPath + "/UMAP/"
    else:
        raise ValueError("parameters.method can only take values 'TSNE' or 'UMAP'")

    if pThreshold is None:
        parameters.pThreshold = [0.33, 0.67]
    else:
        parameters.pThreshold = pThreshold

    zValues = []
    projfiles = glob.glob(tsnefolder + "/*" + endident)
    t1 = time.time()

    zValNames = []
    zValLens = []
    ampVels = []
    for pi, projfile in enumerate(projfiles):
        fname = projfile.split("/")[-1].split(".")[0]
        print(f"Processing {projfile}")
        zValNames.append(fname)
        print(
            "%i/%i Loading embedding for %s %0.02f seconds."
            % (pi + 1, len(projfiles), fname, time.time() - t1)
        )
        with h5py.File(tsnefolder + fname + ".mat", "r") as h5file:
            shape = h5file["trainingEmbedding"][:].T.shape
            print(f"shape of zVals: {shape}")
            print(h5file["trainingEmbedding"][:].T[0:5, :])
            zValues.append(h5file["trainingEmbedding"][:].T)
        ampVels.append(
            np.concatenate(
                ([0], np.linalg.norm(np.diff(zValues[-1], axis=0), axis=1)), axis=0
            )
        )
        # with h5py.File(projectionfolder + fname + '_zAmps_vel.mat', 'r') as h5file:
        #     ampVels.append(h5file['ampvel'][:].T.squeeze())

        assert zValues[-1].shape[0] == ampVels[-1].shape[0]
        zValLens.append(zValues[-1].shape[0])

    zValues = np.concatenate(zValues, 0)
    ampVels = np.concatenate(ampVels, 0)
    # print(zValLens)
    zValLens = np.array(zValLens)
    # print(zValNames)
    zValNames = np.array(zValNames, dtype=object)
    print(f"zValues shape going into watershed: {zValues.shape}")
    # raise Exception("stop here")
    print("Starting watershed transform with AWKDE...")
    LL, wbounds, sigma, xx, density = wshedTransform(
        zValues,
        minimum_regions,
        startsigma,
        tsnefolder,
        saveplot=True,
        training=True,
        use_awkde=False,
        alpha=0.5,
        glob_bw=0.042,
    )

    print("Assigning watershed regions...")
    watershedRegions = np.digitize(zValues, xx.flatten())
    print(f"LL shape: {LL.shape}")
    watershedRegions = LL[watershedRegions[:, 1], watershedRegions[:, 0]]

    if parameters.method == "TSNE":
        print("Calculating velocity distributions...")
        ampVels, pRest = velGMM(
            ampVels,
            parameters,
            parameters.projectPath,
            saveplot=saveplot,
            minimum_regions=minimum_regions,
        )

        outdict = {
            "zValues": zValues,
            "zValNames": zValNames,
            "zValLens": zValLens,
            "sigma": sigma,
            "xx": xx,
            "density": density,
            "LL": LL,
            "watershedRegions": watershedRegions,
            "v": ampVels,
            "pRest": pRest,
            "wbounds": wbounds,
        }
        hdf5storage.write(
            data=outdict,
            path="/",
            truncate_existing=True,
            filename=f"{tsnefolder}{date}_sigma{str(round(sigma,3)).replace('.','_')}_minregions{minimum_regions}_zVals_wShed_groups.mat",
            store_python_metadata=False,
            matlab_compatible=True,
        )

        print("\t tempsave done.")

        t1 = time.time()
        print("Adjusting non-stereotypic regions to 0...")
        bwconn = np.convolve(
            (np.diff(watershedRegions) == 0).astype(bool), np.array([True, True])
        )
        pGoodRest = pRest > np.min(parameters.pThreshold)
        badinds = ~np.bitwise_and(bwconn, pGoodRest)
        watershedRegions[badinds] = 0
        print("\t Done. %0.02f seconds" % (time.time() - t1))
    else:
        pRest = 1.0
    outdict = {
        "zValues": zValues,
        "zValNames": zValNames,
        "zValLens": zValLens,
        "sigma": sigma,
        "xx": xx,
        "density": density,
        "LL": LL,
        "watershedRegions": watershedRegions,
        "v": ampVels,
        "pRest": pRest,
        "wbounds": wbounds,
    }
    hdf5storage.write(
        data=outdict,
        path="/",
        truncate_existing=True,
        filename=f"{tsnefolder}{date}_sigma{str(round(sigma,3)).replace('.','_')}_minregions{minimum_regions}_zVals_wShed_groups.mat",
        store_python_metadata=False,
        matlab_compatible=True,
    )
    print("\t tempsave done.")
