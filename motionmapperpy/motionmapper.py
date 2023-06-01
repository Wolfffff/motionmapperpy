import glob
import multiprocessing as mp
import os
import shutil
import time

import matplotlib

matplotlib.use("Agg")

import pickle
from pathlib import Path

import h5py
import hdf5storage
import matplotlib.pyplot as plt
import numpy as np
from easydict import EasyDict as edict
from scipy.io import loadmat, savemat
from scipy.optimize import fmin
from scipy.spatial import Delaunay, distance
from skimage.filters import roberts
from skimage.segmentation import watershed
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from umap import UMAP
import numpy as np
from scipy.signal import lombscargle

from .mmutils import findPointDensity, gencmap
from .setrunparameters import setRunParameters
from .wavelet import findWavelets

"""Core t-SNE MotionMapper functions."""


def findKLDivergences(data):
    N = len(data)
    logData = np.log(data)
    logData[~np.isfinite(logData)] = 0

    entropies = -np.sum(np.multiply(data, logData), 1)

    D = -np.dot(data, logData.T)

    D = D - entropies[:, None]

    D = D / np.log(2)
    np.fill_diagonal(D, 0)
    return D, entropies


def run_UMAP(data, parameters, save_model=True, metric="euclidean"):
    if not parameters.waveletDecomp:
        raise ValueError("UMAP not implemented without wavelet decomposition.")
    print("Running UMAP with metric: " + parameters.umapMetric)
    # TODO: Determine if want this normalization
    vals = np.sum(data, 1)
    if ~np.all(vals == 1):
        data = data / vals[:, None]

    umapfolder = parameters["projectPath"] + "/UMAP/"
    (
        n_neighbors,
        train_negative_sample_rate,
        min_dist,
        umap_output_dims,
        n_training_epochs,
    ) = (
        parameters["n_neighbors"],
        parameters["train_negative_sample_rate"],
        parameters["min_dist"],
        parameters["umap_output_dims"],
        parameters["n_training_epochs"],
    )

    um = UMAP(
        n_neighbors=n_neighbors,
        negative_sample_rate=train_negative_sample_rate,
        min_dist=min_dist,
        n_components=umap_output_dims,
        n_epochs=n_training_epochs,
        metric=metric,  # TODO: check if this is the right metric
    )
    y = um.fit_transform(data)
    trainmean = np.mean(y, 0)
    scale = parameters["rescale_max"] / np.abs(y).max()
    y = y - trainmean
    y = y * scale

    if save_model:
        print("Saving UMAP model to disk...")
        np.save(
            umapfolder + "_trainMeanScale.npy",
            np.array([trainmean, scale], dtype=object),
        )
        with open(umapfolder + "umap.model", "wb") as f:
            pickle.dump(um, f)

    return y


def run_tSne(data, parameters=None, filename="none"):
    """
    run_tSne runs the t-SNE algorithm on an array of normalized wavelet amplitudes
    :param data: Nxd array of wavelet amplitudes (will normalize if unnormalized) containing N data points
    :param parameters: motionmapperpy Parameters dictionary.
    :return:
            yData -> N x 2 array of embedding results
    """
    parameters = setRunParameters(parameters)

    vals = np.sum(data, 1)
    if ~np.all(vals == 1):
        data = data / vals[:, None]

    if parameters.waveletDecomp:
        print("Finding Distances")
        D, _ = findKLDivergences(data)
        D[~np.isfinite(D)] = 0.0
        D = np.square(D)
        dist_mat_mean = np.mean(D)
        print(f"Distance matrix shape: {D.shape}")
        print(f"Distance matrix mean: {dist_mat_mean}")
        if dist_mat_mean < 0.00001:
            print("Distance matrix mean is too small. Adding to bad file list.")
            with open("list_of_bad_files.txt", "a") as txt_file:
                txt_file.write(f"{filename} \n")

        print("Computing t-SNE with %s method" % parameters.tSNE_method)
        tsne = TSNE(
            perplexity=parameters.perplexity,
            metric="precomputed",
            verbose=1,
            n_jobs=-1,
            method=parameters.tSNE_method,
        )
        yData = tsne.fit_transform(D)
    else:
        print("TSNE fitting complete. Computing Distances")
        tsne = TSNE(
            perplexity=parameters.perplexity,
            learning_rate="auto",
            metric="euclidean",
            verbose=1,
            n_jobs=-1,
            method=parameters.tSNE_method,
            n_iter=parameters.maxOptimIter,
        )
        yData = tsne.fit_transform(data)
        # raise ValueError('tSNE not implemented for runs without wavelet decomposition.')
    return yData


"""Training-set Generation"""


def returnTemplates(yData, signalData, minTemplateLength=10, kdNeighbors=10):
    maxY = np.ceil(np.max(np.abs(yData[:]))) + 1

    nn = NearestNeighbors(n_neighbors=kdNeighbors + 1, n_jobs=-1)
    nn.fit(yData)
    D, _ = nn.kneighbors(yData)
    sigma = np.median(D[:, -1])

    _, xx, density = findPointDensity(yData, sigma, 501, [-maxY, maxY])

    L = watershed(-density, connectivity=10)

    watershedValues = np.digitize(yData, xx)
    watershedValues = L[watershedValues[:, 1], watershedValues[:, 0]]

    maxL = np.max(L)

    templates = []
    for i in range(1, maxL + 1):
        templates.append(signalData[watershedValues == i])
    lengths = np.array([len(i) for i in templates])
    templates = np.array(templates, dtype=object)

    idx = np.where(lengths >= minTemplateLength)[0]
    vals2 = np.zeros(watershedValues.shape)
    for i in range(len(idx)):
        vals2[watershedValues == idx[i] + 1] = i + 1

    templates = templates[lengths >= minTemplateLength]
    lengths = lengths[lengths >= minTemplateLength]

    return templates, xx, density, sigma, lengths, L, vals2


def findTemplatesFromData(
    signalData, yData, signalAmps, numPerDataSet, parameters, projectionFile
):
    kdNeighbors = parameters.kdNeighbors
    minTemplateLength = parameters.minTemplateLength

    print("Finding Templates.")
    templates, _, density, _, templateLengths, L, vals = returnTemplates(
        yData, signalData, minTemplateLength, kdNeighbors
    )

    ####################################################
    wbounds = np.where(roberts(L).astype("bool"))
    wbounds = (wbounds[1], wbounds[0])
    fig, ax = plt.subplots()
    ax.imshow(density, origin="lower", cmap=gencmap())
    ax.scatter(wbounds[0], wbounds[1], color="k", s=0.1)
    fig.savefig(projectionFile[:-4] + "_trainingtSNE.png")
    plt.close()
    print(f"Saved training tSNE plot to {projectionFile[:-4]+'_trainingtSNE.png'}")
    ####################################################

    N = len(templates)
    d = len(signalData[1, :])
    selectedData = np.zeros((numPerDataSet, d))
    selectedAmps = np.zeros((numPerDataSet, 1))

    numInGroup = np.round(numPerDataSet * templateLengths / np.sum(templateLengths))
    numInGroup[numInGroup == 0] = 1
    sumVal = np.sum(numInGroup)
    if sumVal < numPerDataSet:
        q = int(numPerDataSet - sumVal)
        idx = np.random.permutation(N)[: min(q, N)]
        numInGroup[idx] = numInGroup[idx] + 1
    else:
        if sumVal > numPerDataSet:
            q = int(sumVal - numPerDataSet)
            idx2 = np.where(numInGroup > 1)[0]
            Lq = len(idx2)
            if Lq < q:
                idx2 = np.arange(len(numInGroup))
            idx = np.random.permutation(len(idx2))[:q]
            numInGroup[idx2[idx]] = numInGroup[idx2[idx]] - 1
    idx = numInGroup > templateLengths
    numInGroup[idx] = templateLengths[idx]
    cumSumGroupVals = [0] + np.cumsum(numInGroup).astype(int).tolist()

    for j in range(N):
        if cumSumGroupVals[j + 1] > cumSumGroupVals[j]:
            amps = signalAmps[vals == j + 1]
            idx2 = np.random.permutation(len(templates[j][:, 1]))[
                : int(numInGroup[j])
            ].astype(int)
            selectedData[cumSumGroupVals[j] : cumSumGroupVals[j + 1], :] = templates[j][
                idx2, :
            ]
            selectedAmps[cumSumGroupVals[j] : cumSumGroupVals[j + 1], 0] = amps[idx2]

    signalData = selectedData
    signalAmps = selectedAmps

    return signalData, signalAmps


def mm_findWavelets(projections, numModes, parameters):
    amplitudes, f = findWavelets(
        projections,
        numModes,
        parameters.omega0,
        parameters.numPeriods,
        parameters.samplingFreq,
        parameters.maxF,
        parameters.minF,
        parameters.numProcessors,
        parameters.useGPU,
    )
    return amplitudes, f


import pathlib


def file_embeddingSubSampling(projectionFile, parameters):
    perplexity = parameters.training_perplexity

    if parameters.waveletDecomp:
        print("\n Loading wavelets")
        # projections = np.array(loadmat(projectionFile, variable_names=['projections'])['projections'])
        with h5py.File(
            f"{parameters.projectPath}/Subsampled_wavelets/{pathlib.Path(projectionFile).stem}-subsampled-wavelets.mat",
            "r",
        ) as f:
            data = f["signaldata"][:]  # [signalIdx]

        print(f"Data shape: {data.shape}")
        print("\n Loaded wavelets")
        # data = loadmat(f'{parameters.projectPath}/Wavelets/{pathlib.Path(projectionFile).stem}-wavelets.mat')

        print("\n Subsampled wavelets")

        signalData = data

    print("\n Subsampled projections")
    signalAmps = np.sum(signalData, axis=1)

    signalData = signalData / signalAmps[:, None]

    if parameters.method == "TSNE":
        parameters.perplexity = perplexity
        yData = run_tSne(signalData, parameters, projectionFile)
    elif parameters.method == "UMAP":
        yData = run_UMAP(
            signalData,
            parameters,
            save_model=False,
            metric=parameters.umapSubsampMetric,
        )
    else:
        raise ValueError("Supported parameter.method are 'TSNE' or 'UMAP'")
    return yData, signalData, np.arange(parameters.training_numPoints), signalAmps


from tqdm import tqdm


def get_wavelets(projectionFiles, parameters, i, ls=False):
    # L = len(projectionFiles)
    # for i in tqdm(range(L)):
    print(f"Processing {projectionFiles[i]}")
    if ls:
        calc_and_write_wavelets_ls(projectionFiles[i], parameters)
    else:
        calc_and_write_wavelets(projectionFiles[i], parameters)


def mm_findWavelets_ls(projections, parameters):
    t1 = time.time()
    print("\t Calculating wavelets, clock starting.")

    import multiprocessing as mp
    import numpy as np

    if parameters.numProcessors < 0:
        parameters.numProcessors = mp.cpu_count()
    print("\t Using #%i CPUs." % parameters.numProcessors)
    print("Using Lomb-Scargle.")

    projections = np.array(projections)
    t1 = time.time()

    minT = 1.0 / parameters.maxF
    maxT = 1.0 / parameters.minF
    Ts = minT * (
        2
        ** (
            (np.arange(parameters.numPeriods) * np.log(maxT / minT))
            / (np.log(2) * (parameters.numPeriods - 1))
        )
    )
    f = (1.0 / Ts)[::-1]

    # TODO: Move this to parameters
    omega0 = 20
    scales = (omega0 + np.sqrt(2 + omega0**2)) / (4 * np.pi * f)
    window_sizes = np.round(scales * parameters.samplingFreq).astype(int)
    print(f"Window sizes: {window_sizes}")
    print(f"Frequencies: {f}, shape: {f.shape}")

    N = projections.shape[0]
    print(f"Projection shape: {projections.shape}")
    print("No normalization -- precentering though.")
    try:
        pool = mp.Pool(parameters.numProcessors)
        print(f"Scarglin' {projections.shape[1]} projections")
        amplitudes = pool.starmap(
            rolling_lombscargle,
            [
                (
                    projections[:, i],
                    np.linspace(0, N / parameters.samplingFreq, N),
                    f.astype(float),
                    window_sizes,
                )
                for i in range(projections.shape[1])
            ],
        )
        amplitudes = np.concatenate(amplitudes, 0)
        amplitudes[~np.isfinite(amplitudes)] = 0
        print(f"Done Scarglin' -- shape: {amplitudes.shape}")
        pool.close()
        pool.join()
    except Exception as E:
        pool.close()
        pool.join()
        raise E
    print("\t Done at %0.02f seconds." % (time.time() - t1))
    return amplitudes.T, f, window_sizes


def rolling_window_with_padding(arr, window_size):
    # TODO: double check this
    padding = (window_size - 1) // 2
    padded_arr = np.pad(arr, (padding, padding), mode="edge")
    shape = padded_arr.shape[:-1] + (
        padded_arr.shape[-1] - window_size + 1,
        window_size,
    )
    strides = padded_arr.strides + (padded_arr.strides[-1],)

    return np.lib.stride_tricks.as_strided(padded_arr, shape=shape, strides=strides)


def rolling_lombscargle(data, sampling_times, freqs, window_sizes):
    # print(f"Inside rolling_lombscargle - data shape: {data.shape}")  # Debug print
    # print(
    #     f"Inside rolling_lombscargle - sampling_times shape: {sampling_times.shape}"
    # )  # Debug print
    # print(f"Inside rolling_lombscargle - freqs shape: {freqs.shape}")  # Debug print

    # Initialize an empty array to store the Lomb-Scargle periodograms
    periodograms = np.zeros((data.size, freqs.size))

    # Loop through each frequency and its corresponding window size
    for f_idx, (freq, window_size) in enumerate(zip(freqs, window_sizes)):
        print(f"On frequency {f_idx} of {freqs.size}")
        # print(
        #     f"Inside rolling_lombscargle -  freq: {freq}, win size: {window_size}"
        # )  # Debug print
        # print(f"Inside rolling_lombscargle -  freq: {freq}")  # Debug print
        windows = rolling_window_with_padding(data, window_size)
        # print(
        #     f"Inside rolling_lombscargle -  windows shape: {windows.shape}"
        # )  # Debug print
        windows_sampling_times = rolling_window_with_padding(
            sampling_times, window_size
        )

        for i, (window, times) in enumerate(zip(windows, windows_sampling_times)):
            angular_frequency = 2 * np.pi * freq
            tmp_window = window.copy()
            # print(f"Inside rolling_lombscargle -  window shape: {window.shape}")
            window = window[np.isfinite(tmp_window)]
            # print(f"Post nan removal -  window shape: {window.shape}")
            # sampling_times = sampling_times[np.isfinite(tmp_window)]
            sampling_times_window = times[np.isfinite(tmp_window)]
            # print(f"Processing window {i} of {windows.shape[0]}")
            periodogram = lombscargle(
                sampling_times_window,
                window,
                [angular_frequency],
                normalize=False,
                precenter=True,
            )

            if np.all(np.isnan(periodogram)):
                periodogram = 0

            periodograms[i, f_idx] = periodogram
    return periodograms.T


def calc_and_write_wavelets_ls(projectionFile, parameters):
    # calculate and write wavelets with lomb-scargle from scipy
    print("\t Loading Projections")

    with h5py.File(projectionFile, "r") as hfile:
        projections = hfile["projections"][:].T
    projections = np.array(projections)

    if parameters.waveletDecomp:
        print("\t Calculating Wavelets")
        if not os.path.exists(
            f"{parameters.projectPath}/Wavelets/{pathlib.Path(projectionFile).stem}-wavelets.mat"
        ):
            data, freqs, win_sizes = mm_findWavelets_ls(projections, parameters)
            print(f"\n Saving wavelets: {data.shape}")
            with h5py.File(
                f"{parameters.projectPath}/Wavelets/{pathlib.Path(projectionFile).stem}-wavelets.mat",
                "w",
                libver="latest",
            ) as f:
                print("No compression")
                f.create_dataset("wavelets", data=data)
                f.create_dataset("f", data=freqs)
                f.create_dataset("win_sizes", data=win_sizes)


def calc_and_write_wavelets(projectionFile, parameters):
    print("\t Loading Projections")

    with h5py.File(projectionFile, "r") as hfile:
        projections = hfile["projections"][:].T
    projections = np.array(projections)

    if parameters.waveletDecomp:
        print("\t Calculating Wavelets")
        if not os.path.exists(
            f"{parameters.projectPath}/Wavelets/{pathlib.Path(projectionFile).stem}-wavelets.mat"
        ):
            data, freqs = mm_findWavelets(projections, parameters.pcaModes, parameters)
            print(f"\n Saving wavelets: {data.shape}")
            with h5py.File(
                f"{parameters.projectPath}/Wavelets/{pathlib.Path(projectionFile).stem}-wavelets.mat",
                "w",
                libver="latest",
            ) as f:
                print("No compression")
                f.create_dataset("wavelets", data=data)
                f.create_dataset("f", data=freqs)


import natsort

from tqdm import tqdm


def runEmbeddingSubSampling(projectionDirectory, parameters):
    """
    runEmbeddingSubSampling generates a training set given a set of .mat files.

    :param projectionDirectory: directory path containing .mat projection files.
    Each of these files should contain an N x pcaModes variable, 'projections'.
    :param parameters: motionmapperpy Parameters dictionary.
    :return:
        trainingSetData -> normalized wavelet training set
                           (N x (pcaModes*numPeriods) )
        trainingSetAmps -> Nx1 array of training set wavelet amplitudes
        projectionFiles -> list of files in 'projectionDirectory'
    """
    parameters = setRunParameters(parameters)
    projectionFiles = glob.glob(projectionDirectory + "/*pcaModes.mat")
    projectionFiles = natsort.natsorted(projectionFiles)
    for projectionFile in projectionFiles.copy():
        print(f"Checking {projectionFile}")
        if not os.path.exists(
            f"{parameters.projectPath}/Subsampled_wavelets/{pathlib.Path(projectionFile).stem}-subsampled-wavelets.mat"
        ):
            print(f"Skipping {projectionFile}")
            projectionFiles.remove(projectionFile)
    N = parameters.trainingSetSize
    L = len(projectionFiles)

    numPerDataSet = round(N / L)
    print(f"Number of files: {L}")
    print(f"Number of samples per file: {numPerDataSet}")
    numModes = parameters.pcaModes
    numPeriods = parameters.numPeriods

    if numPerDataSet > parameters.training_numPoints:
        raise ValueError(
            "miniTSNE size is %i samples per file which is low for current trainingSetSize which "
            "requries %i samples per file. "
            "Please decrease trainingSetSize or increase training_numPoints."
            % (parameters.training_numPoints, numPerDataSet)
        )

    if parameters.waveletDecomp:
        trainingSetData = np.zeros((numPerDataSet * L, numModes * numPeriods))
    else:
        trainingSetData = np.zeros((numPerDataSet * L, numModes))
    trainingSetAmps = np.zeros((numPerDataSet * L, 1))
    useIdx = np.ones((numPerDataSet * L), dtype="bool")

    for i in tqdm(range(L)):
        print(
            "Finding training set contributions from data set %i/%i : \n%s"
            % (i + 1, L, projectionFiles[i])
        )

        currentIdx = np.arange(numPerDataSet) + (i * numPerDataSet)

        yData, signalData, _, signalAmps = file_embeddingSubSampling(
            projectionFiles[i], parameters
        )
        (
            trainingSetData[currentIdx, :],
            trainingSetAmps[currentIdx],
        ) = findTemplatesFromData(
            signalData, yData, signalAmps, numPerDataSet, parameters, projectionFiles[i]
        )

        a = np.sum(trainingSetData[currentIdx, :], 1) == 0
        useIdx[currentIdx[a]] = False

    trainingSetData = trainingSetData[useIdx, :]
    trainingSetAmps = trainingSetAmps[useIdx]

    return trainingSetData, trainingSetAmps, projectionFiles


def subsampled_tsne_from_projections(parameters, results_directory):
    """
    Wrapper function for training set subsampling and mapping.
    """
    projection_directory = results_directory + "/Projections/"
    if parameters.method == "TSNE":
        if parameters.waveletDecomp:
            tsne_directory = results_directory + "/TSNE/"
        else:
            tsne_directory = results_directory + "/TSNE_Projections/"

        parameters.tsne_directory = tsne_directory

        parameters.tsne_readout = 50

        tSNE_method_old = parameters.tSNE_method
        if tSNE_method_old != "barnes_hut":
            print(
                "Setting tsne method to barnes_hut while subsampling for training set (for speedup)..."
            )
            parameters.tSNE_method = "barnes_hut"

    elif parameters.method == "UMAP":
        tsne_directory = results_directory + "/UMAP/"
        if not parameters.waveletDecomp:
            raise ValueError("Wavelet decomposition needed to run UMAP implementation.")
    else:
        raise ValueError("Supported parameter.method are 'TSNE' or 'UMAP'")

    print("Finding Training Set")
    if not os.path.exists(tsne_directory + "training_data.mat"):
        trainingSetData, trainingSetAmps, _ = runEmbeddingSubSampling(
            projection_directory, parameters
        )
        if os.path.exists(tsne_directory):
            shutil.rmtree(tsne_directory)
            os.mkdir(tsne_directory)
        else:
            os.mkdir(tsne_directory)

        hdf5storage.write(
            data={"trainingSetData": trainingSetData},
            path="/",
            truncate_existing=True,
            filename=tsne_directory + "/training_data.mat",
            store_python_metadata=False,
            matlab_compatible=True,
        )

        hdf5storage.write(
            data={"trainingSetAmps": trainingSetAmps},
            path="/",
            truncate_existing=True,
            filename=tsne_directory + "/training_amps.mat",
            store_python_metadata=False,
            matlab_compatible=True,
        )

        del trainingSetAmps
    else:
        print(
            "Subsampled trainingSetData found, skipping minitSNE and running training tSNE"
        )
        with h5py.File(tsne_directory + "/training_data.mat", "r") as hfile:
            trainingSetData = hfile["trainingSetData"][:].T

    # %% Run t-SNE on training set
    if parameters.method == "TSNE":
        if tSNE_method_old != "barnes_hut":
            print("Setting tsne method back to to %s" % tSNE_method_old)
            parameters.tSNE_method = tSNE_method_old
        parameters.tsne_readout = 5
        print("Finding t-SNE Embedding for Training Set")

        trainingEmbedding = run_tSne(trainingSetData, parameters)
    elif parameters.method == "UMAP":
        print("Finding UMAP Embedding for Training Set")
        trainingEmbedding = run_UMAP(
            trainingSetData, parameters, metric=parameters.umapMetric
        )
    else:
        raise ValueError("Supported parameter.method are 'TSNE' or 'UMAP'")
    hdf5storage.write(
        data={"trainingEmbedding": trainingEmbedding},
        path="/",
        truncate_existing=True,
        filename=tsne_directory + "/training_embedding.mat",
        store_python_metadata=False,
        matlab_compatible=True,
    )


"""Re-Embedding Code"""


def returnCorrectSigma_sparse(ds, perplexity, tol, maxNeighbors):
    highGuess = np.max(ds)
    lowGuess = 1e-12

    sigma = 0.5 * (highGuess + lowGuess)

    dsize = ds.shape
    sortIdx = np.argsort(ds)
    ds = ds[sortIdx[:maxNeighbors]]
    p = np.exp(-0.5 * np.square(ds) / sigma**2)
    p = p / np.sum(p)
    idx = p > 0
    H = np.sum(-np.multiply(p[idx], np.log(p[idx])) / np.log(2))
    P = 2**H

    if abs(P - perplexity) < tol:
        test = False
    else:
        test = True

    count = 0
    if ~np.isfinite(sigma):
        raise ValueError(
            "Starting sigma is %0.02f, highGuess is %0.02f "
            "and lowGuess is %0.02f" % (sigma, highGuess, lowGuess)
        )
    while test:
        if P > perplexity:
            highGuess = sigma
        else:
            lowGuess = sigma

        sigma = 0.5 * (highGuess + lowGuess)

        p = np.exp(-0.5 * np.square(ds) / sigma**2)
        if np.sum(p) > 0:
            p = p / np.sum(p)
        idx = p > 0
        H = np.sum(-np.multiply(p[idx], np.log(p[idx])) / np.log(2))
        P = 2**H

        if np.abs(P - perplexity) < tol:
            test = False

    out = np.zeros((dsize[0],))
    out[sortIdx[:maxNeighbors]] = p
    return sigma, out


def findListKLDivergences(data, data2):
    logData = np.log(data)

    entropies = -np.sum(np.multiply(data, logData), 1)
    del logData

    logData2 = np.log(data2)

    D = -np.dot(data, logData2.T)

    D = D - entropies[:, None]

    D = D / np.log(2)
    return D, entropies


def calculateKLCost(x, ydata, ps):
    d = np.sum(np.square(ydata - x), 1).T
    out = np.log(np.sum(1 / (1 + d))) + np.sum(np.multiply(ps, np.log(1 + d)))
    return out


def TDistProjs(
    i,
    q,
    perplexity,
    sigmaTolerance,
    maxNeighbors,
    trainingEmbedding,
    readout,
    waveletDecomp,
):
    if (i + 1) % readout == 0:
        t1 = time.time()
        print("\t\t Calculating Sigma Image #%5i" % (i + 1))
    _, p = returnCorrectSigma_sparse(q, perplexity, sigmaTolerance, maxNeighbors)

    if (i + 1) % readout == 0:
        print("\t\t Calculated Sigma Image #%5i" % (i + 1))

    idx2 = p > 0
    z = trainingEmbedding[idx2, :]
    maxIdx = np.argmax(p)
    a = np.sum(z * (p[idx2].T)[:, None], axis=0)

    guesses = [a, trainingEmbedding[maxIdx, :]]

    q = Delaunay(z)

    if (i + 1) % readout == 0:
        print("\t\t FminSearch Image #%5i" % (i + 1))

    b = np.zeros((2, 2))
    c = np.zeros((2,))
    flags = np.zeros((2,))

    if waveletDecomp:
        costfunc = calculateKLCost
    else:
        costfunc = calculateKLCost

    b[0, :], c[0], _, _, flags[0] = fmin(
        costfunc,
        x0=guesses[0],
        args=(z, p[idx2]),
        disp=False,
        full_output=True,
        maxiter=100,
    )
    b[1, :], c[1], _, _, flags[1] = fmin(
        costfunc,
        x0=guesses[1],
        args=(z, p[idx2]),
        disp=False,
        full_output=True,
        maxiter=100,
    )
    if (i + 1) % readout == 0:
        print(
            "\t\t FminSearch Done Image #%5i %0.02fseconds \n Flags are %s"
            % (i + 1, time.time() - t1, flags)
        )

    polyIn = q.find_simplex(b) >= 0

    if np.sum(polyIn) > 0:
        pp = np.where(polyIn)[0]
        mI = np.argmin(c[polyIn])
        mI = pp[mI]
        current_poly = True
    else:
        mI = np.argmin(c)
        current_poly = False
    if (i + 1) % readout == 0:
        print(
            "\t\t Simplex search done Image #%5i %0.02fseconds"
            % (i + 1, time.time() - t1)
        )
    exitFlags = flags[mI]
    current_guesses = guesses[mI]
    current = b[mI]
    tCosts = c[mI]
    current_meanMax = mI
    return current_guesses, current, tCosts, current_poly, current_meanMax, exitFlags


def findTDistributedProjections_fmin(data, trainingData, trainingEmbedding, parameters):
    readout = 100000
    sigmaTolerance = 1e-5
    perplexity = parameters.perplexity
    maxNeighbors = parameters.maxNeighbors
    batchSize = parameters.embedding_batchSize

    N = len(data)
    zValues = np.zeros((N, 2))
    zGuesses = np.zeros((N, 2))
    zCosts = np.zeros((N,))
    batches = np.ceil(N / batchSize).astype(int)
    inConvHull = np.zeros((N,), dtype=bool)
    meanMax = np.zeros((N,))
    exitFlags = np.zeros((N,))

    if parameters.numProcessors < 0:
        numProcessors = mp.cpu_count()
    else:
        numProcessors = parameters.numProcessors
    # ctx = mp.get_context('spawn')

    for j in range(batches):
        print("\t Processing batch #%4i out of %4i" % (j + 1, batches))
        idx = np.arange(batchSize) + j * batchSize
        idx = idx[idx < N]
        currentData = data[idx, :]

        if parameters.waveletDecomp:
            if np.sum(currentData == 0):
                print(
                    "Zeros found in wavelet data at following positions. Will replace then with 1e-12."
                )
                currentData[currentData == 0] = 1e-12

            print("\t Calculating distances for batch %4i" % (j + 1))
            t1 = time.time()
            D2, _ = findListKLDivergences(currentData, trainingData)
            print(
                "\t Calculated distances for batch %4i %0.02fseconds."
                % (j + 1, time.time() - t1)
            )
        else:
            print("\t Calculating distances for batch %4i" % (j + 1))
            t1 = time.time()
            D2 = distance.cdist(currentData, trainingData, metric="sqeuclidean")
            print(
                "\t Calculated distances for batch %4i %0.02fseconds."
                % (j + 1, time.time() - t1)
            )

        print("\t Calculating fminProjections for batch %4i" % (j + 1))
        t1 = time.time()
        pool = mp.Pool(numProcessors)
        outs = pool.starmap(
            TDistProjs,
            [
                (
                    i,
                    D2[i, :],
                    perplexity,
                    sigmaTolerance,
                    maxNeighbors,
                    trainingEmbedding,
                    readout,
                    parameters.waveletDecomp,
                )
                for i in range(len(idx))
            ],
        )

        zGuesses[idx, :] = np.concatenate([out[0][:, None] for out in outs], axis=1).T
        zValues[idx, :] = np.concatenate([out[1][:, None] for out in outs], axis=1).T
        zCosts[idx] = np.array([out[2] for out in outs])
        inConvHull[idx] = np.array([out[3] for out in outs])
        meanMax[idx] = np.array([out[4] for out in outs])
        exitFlags[idx] = np.array([out[5] for out in outs])
        pool.close()
        pool.join()
        print(
            "\t Processed batch #%4i out of %4i in %0.02fseconds.\n"
            % (j + 1, batches, time.time() - t1)
        )

    zValues[~inConvHull, :] = zGuesses[~inConvHull, :]

    return zValues, zCosts, zGuesses, inConvHull, meanMax, exitFlags


import multiprocessing


def findEmbeddings(
    projections, trainingData, trainingEmbedding, parameters, projectionFile
):
    """
    findEmbeddings finds the optimal embedding of a data set into a previously
    found t-SNE embedding.
    :param projections:  N x (pcaModes x numPeriods) array of projection values.
    :param trainingData: Nt x (pcaModes x numPeriods) array of wavelet amplitudes containing Nt data points.
    :param trainingEmbedding: Nt x 2 array of embeddings.
    :param parameters: motionmapperpy Parameters dictionary.
    :return: zValues : N x 2 array of embedding results, outputStatistics : dictionary containing other parametric
    outputs.
    """
    d = projections.shape[1]
    numModes = parameters.pcaModes
    numPeriods = parameters.numPeriods

    if parameters.waveletDecomp:
        print("Finding Wavelets")
        if not os.path.exists(
            f"{parameters.projectPath}/Wavelets/{pathlib.Path(projectionFile).stem}-wavelets.mat"
        ):
            data, f = mm_findWavelets(projections, numModes, parameters)
        else:
            print("\n Loading wavelets")
            with h5py.File(
                f"{parameters.projectPath}/Wavelets/{pathlib.Path(projectionFile).stem}-wavelets.mat",
                "r",
            ) as f:
                data = f["wavelets"][:]
                data[~np.isfinite(data)] = 1e-12
                data[data == 0] = 1e-12
        print("\n Loaded wavelets")
    else:
        print("Using projections for tSNE. No wavelet decomposition.")
        f = 0
        data = projections

    raw_wavelets = data.copy()
    # Apply condition on wavelets
    data_sum = np.sum(raw_wavelets, 1)
    idx_valid = data_sum > 4e2  # indices of valid points

    # Only valid points are embedded
    valid_data = data[idx_valid]
    print(f"Valid data shape: {valid_data.shape} out of {data.shape}")

    data = data / np.sum(data, 1)[:, None]

    print("Finding Embeddings")
    t1 = time.time()
    if parameters.method == "TSNE":
        (
            zValues_temp,
            zCosts,
            zGuesses,
            inConvHull,
            meanMax,
            exitFlags,
        ) = findTDistributedProjections_fmin(
            valid_data, trainingData, trainingEmbedding, parameters
        )

        outputStatistics_temp = edict()
        outputStatistics_temp.zCosts = zCosts
        outputStatistics_temp.f = f
        outputStatistics_temp.numModes = numModes
        outputStatistics_temp.zGuesses = zGuesses
        outputStatistics_temp.inConvHull = inConvHull
        outputStatistics_temp.meanMax = meanMax
        outputStatistics_temp.exitFlags = exitFlags
    elif parameters.method == "UMAP":
        # Split valid data into chunks for parallel processing

        pool = multiprocessing.Pool(processes=parameters.numProcessors)
        print(f"Using {parameters.numProcessors} processors for embedding")
        valid_data_chunks = np.array_split(valid_data, parameters.numProcessors)

        # Parallelize the UMAP transform for valid data chunks
        results = pool.starmap(
            umap_transform, [(chunk, parameters) for chunk in valid_data_chunks]
        )

        # Combine the results
        zValues_temp_list, trainparams_list = zip(*results)
        zValues_temp = np.concatenate(zValues_temp_list)
        trainparams = trainparams_list[
            0
        ]  # Assuming trainparams are the same for all chunks

        outputStatistics = edict()
        outputStatistics.training_mean = trainparams[0]
        outputStatistics.training_scale = trainparams[1]
    else:
        raise ValueError("Supported parameter.method are 'TSNE' or 'UMAP'")

        # Initialize zValues with 'NA' for all points
    zValues = np.full((data.shape[0], 2), np.nan)

    # Assign computed embeddings to valid points
    zValues[idx_valid] = zValues_temp

    del data
    print("Embeddings found in %0.02f seconds." % (time.time() - t1))

    return zValues, outputStatistics


def umap_transform(data, parameters):
    umapfolder = parameters["projectPath"] + "/UMAP/"
    with open(umapfolder + "umap.model", "rb") as f:
        um = pickle.load(f)
    trainparams = np.load(umapfolder + "_trainMeanScale.npy", allow_pickle=True)
    embed_negative_sample_rate = parameters["embed_negative_sample_rate"]
    um.negative_sample_rate = embed_negative_sample_rate
    zValues = um.transform(data)
    zValues = zValues - trainparams[0]
    zValues = zValues * trainparams[1]
    return zValues, trainparams


def file_embeddingSubSampling_batch(projectionFile, parameters):
    numPoints = parameters.training_numPoints

    with h5py.File(projectionFile, "r") as hfile:
        projections_shape = hfile["projections"][:].T.shape
    # TODO: Make this more general -- should get the others if we aren't using samples
    edge_file = (
        "/Genomics/ayroleslab2/scott/git/lts-manuscript/analysis/sample_tracks/edge/"
        + pathlib.Path(projectionFile).stem.split("-")[0]
        + "_edge.mat"
    )
    print(f"Using edge file: {edge_file}")

    # fly_num = int(pathlib.Path(projectionFile).stem.split("-")[3].split("_")[0])
    with h5py.File(edge_file, "r") as hfile:
        edge_mask = np.append([False], hfile["edger"][:].T[:, 0].astype(bool))
        # edge_mask = np.append([False], hfile["edger"][:].T[:, fly_num].astype(bool))
    print(f"projection file: {projectionFile}")
    print(f"edge file: {edge_file}")

    print(f"projections shape: {projections_shape}")
    print(f"edge shape: {edge_mask.shape}")
    # edge_mask = edge_mask[: projections_shape[0]]
    print(f"Frac on edge: {np.sum(edge_mask)/projections_shape[0]}")

    missingness_file = f"{parameters.projectPath}/Ego/{Path(projectionFile).stem}.h5"
    with h5py.File(
        missingness_file,
        "r",
    ) as hfile:
        print(f"projection file: {projectionFile}")
        print(f"missingness file: {missingness_file}")
        missingness_mask = hfile["missing_data_indices"][:].T.astype(bool)

        print(f"projections shape: {missingness_mask.shape}")
        print(f"missingness shape: {edge_mask.shape}")
        print(f"Frac missing: {np.sum(missingness_mask)/projections_shape[0]}")
    if projections_shape[0] < numPoints:
        raise ValueError(
            "Training number of points for miniTSNE is greater than # samples in some files. Please "
            "adjust it to %i or lower" % (projections_shape[0])
        )

    N = projections_shape[0]
    skipLength = np.floor(N / numPoints).astype(int)
    if skipLength == 0:
        skipLength = 1
        numPoints = N

    print(f"Subsampling {N} points to {numPoints} points")
    # 20230514-mmpy-lts-all-pchip5-headprobinterpy0xhead-medianwin5-gaussian-lombscargle-dynamicwinomega020-singleflysampledtracks
    # cp -r /Gtt/git/lts-manuscript/analysis/20230514-mmpy-lts-all-pchip5-headprobinterpy0xhead-medianwin5-gaussian-lombscargle-dynamicwinomega020-singleflysampledtracks/Wavelets ./
    if parameters.waveletDecomp:
        # TODO: Don't be stupid. Load the wavelets once and then subsample them.
        with h5py.File(
            f"{parameters.projectPath}/Wavelets/{pathlib.Path(projectionFile).stem}-wavelets.mat",
            "r",
        ) as f:
            wlets = f["wavelets"][:]
            print(f"wavelets shape: {wlets.shape}")
            sum_mask = np.sum(wlets, axis=1) < 4e2
            print(f"sum_mask shape: {sum_mask.shape}")
            print(
                f"Fraction with amp lower than 4e2: {np.sum(sum_mask)/projections_shape[0]}"
            )
        signalIdx = np.indices((projections_shape[0],))[0]
        print(f"signalIdx shape: {signalIdx.shape}")
        print(f"edge_mask shape: {edge_mask.shape}")
        mask = np.any(np.vstack([edge_mask, missingness_mask, sum_mask]).T, axis=1)
        print(f"mask shape: {mask.shape}")
        signalIdx = signalIdx[[not mask_ele for mask_ele in mask]]
        # Subset to remove edge calls
        if signalIdx.shape[0] < numPoints:
            print("Warning: Not enough points to sample. Using all points")
            if skipLength == 0:
                skipLength = 1
                numPoints = signalIdx.shape[0]
            print(f"Final signalIdx.shape: {signalIdx.shape}")
        else:
            print(f"Found {signalIdx.shape[0]} points to sample")
            skipLength = np.floor(signalIdx.shape[0] / numPoints).astype(int)
            signalIdx = signalIdx[0 : int(0 + (numPoints) * skipLength) : skipLength]
        print(f"Final signalIdx: {signalIdx[0:10]}")
        print(f"Final signalIdx.shape: {signalIdx.shape}")
        print("\t Calculating Wavelets")
        print(
            f"{parameters.projectPath}/Wavelets/{pathlib.Path(projectionFile).stem}-wavelets.mat"
        )
        print("\n Loading wavelets")
        with h5py.File(
            f"{parameters.projectPath}/Wavelets/{pathlib.Path(projectionFile).stem}-wavelets.mat",
            "r",
        ) as f:
            data = f["wavelets"][sorted(signalIdx)]
        print("\n Loaded wavelets")

        # get templates and real training data

        with open("list_of_working_files.txt", "a") as myfile:
            myfile.write(f"{projectionFile}\n")
        with open("list_of_working_files_length.txt", "a") as myfile:
            myfile.write(f"{signalIdx.shape[0]}\n")

        print("\n Subsampled wavelets")
        if not os.path.exists(
            f"{parameters.projectPath}/Subsampled_wavelets/{pathlib.Path(projectionFile).stem}-subsampled-wavelets.mat"
        ):
            with h5py.File(
                f"{parameters.projectPath}/Subsampled_wavelets/{pathlib.Path(projectionFile).stem}-subsampled-wavelets.mat",
                "w",
            ) as f:
                f.create_dataset("signaldata", data=data, compression="lzf")
        else:
            print("File already exists")
    return


def runEmbeddingSubSampling_batch(projectionDirectory, parameters, i):
    """
    runEmbeddingSubSampling generates a training set given a set of .mat files.

    :param projectionDirectory: directory path containing .mat projection files.
    Each of these files should contain an N x pcaModes variable, 'projections'.
    :param parameters: motionmapperpy Parameters dictionary.
    :return:
        trainingSetData -> normalized wavelet training set
                           (N x (pcaModes*numPeriods) )
        trainingSetAmps -> Nx1 array of training set wavelet amplitudes
        projectionFiles -> list of files in 'projectionDirectory'
    """
    parameters = setRunParameters(parameters)
    projectionFiles = glob.glob(projectionDirectory + "/*pcaModes.mat")
    projectionFiles = natsort.natsorted(projectionFiles)

    file_embeddingSubSampling_batch(projectionFiles[i], parameters)


def subsampled_tsne_from_projections_batch(parameters, results_directory, i):
    """
    Wrapper function for training set subsampling and mapping.
    """
    projection_directory = results_directory + "/Projections/"
    if parameters.method == "TSNE":
        if parameters.waveletDecomp:
            tsne_directory = results_directory + "/TSNE/"
        else:
            tsne_directory = results_directory + "/TSNE_Projections/"

        parameters.tsne_directory = tsne_directory

        parameters.tsne_readout = 50

        tSNE_method_old = parameters.tSNE_method
        if tSNE_method_old != "barnes_hut":
            print(
                "Setting tsne method to barnes_hut while subsampling for training set (for speedup)..."
            )
            parameters.tSNE_method = "barnes_hut"

    elif parameters.method == "UMAP":
        tsne_directory = results_directory + "/UMAP/"
        if not parameters.waveletDecomp:
            raise ValueError("Wavelet decomposition needed to run UMAP implementation.")
    else:
        raise ValueError("Supported parameter.method are 'TSNE' or 'UMAP'")

    print("Finding Training Set")
    # if not os.path.exists(tsne_directory + "training_data.mat"):
    runEmbeddingSubSampling_batch(projection_directory, parameters, i)
