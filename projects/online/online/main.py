import logging
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import torch
import torch.multiprocessing as mp
from amplfi.train.architectures.embeddings import MultiModalPsd
from amplfi.train.architectures.flows import MaskedAutoregressiveFlow
from amplfi.train.architectures.flows.base import FlowArchitecture
from architectures import Architecture
from ml4gw.nn.norm import GroupNorm1DGetter
from ml4gw.transforms import ChannelWiseScaler, SpectralDensity, Whiten
from torch.multiprocessing import Process, Queue

from ledger.events import EventSet
from online.utils.buffer import InputBuffer, OutputBuffer
from online.utils.dataloading import data_iterator
from online.utils.gdb import GdbServer, GraceDb, authenticate, gracedb_factory
from online.utils.ngdd import data_iterator as ngdd_data_iterator
from online.utils.pastro import fit_or_load_pastro
from online.utils.pe import run_amplfi
from online.utils.searcher import Searcher
from online.utils.snapshotter import OnlineSnapshotter
from utils.preprocessing import BatchWhitener

if TYPE_CHECKING:
    from pastro.pastro import Pastro

SECONDS_PER_DAY = 86400


def load_model(model: Architecture, weights: Path):
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    arch_weights = {
        k[6:]: v
        for k, v in checkpoint["state_dict"].items()
        if k.startswith("model.")
    }
    model.load_state_dict(arch_weights)
    model.eval()
    return model, checkpoint


def load_amplfi(model: FlowArchitecture, weights: Path, num_params: int):
    model, checkpoint = load_model(model, weights)
    scaler_weights = {
        k[len("scaler.") :]: v
        for k, v in checkpoint["state_dict"].items()
        if k.startswith("scaler.")
    }
    scaler = ChannelWiseScaler(num_params)
    scaler.load_state_dict(scaler_weights)
    return model, scaler


def get_time_offset(
    inference_sampling_rate: float,
    fduration: float,
    integration_window_length: float,
    kernel_length: float,
    aframe_right_pad: float,
):
    # offset the initial timestamp of our
    # integrated outputs relative to the
    # initial timestamp of the most recently
    # loaded frames
    time_offset = (
        1 / inference_sampling_rate  # end of the first kernel in batch
        - fduration / 2  # account for whitening padding
        - integration_window_length  # account for time to build peak
    )

    if aframe_right_pad > 0:
        time_offset -= kernel_length - aframe_right_pad
    elif aframe_right_pad < 0:
        # Trigger distance parameter accounts for fduration already
        time_offset -= abs(aframe_right_pad) - fduration / 2

    return time_offset


@torch.no_grad()
def search(
    whitener: BatchWhitener,
    snapshotter: OnlineSnapshotter,
    searcher: Searcher,
    event_queue: Queue,
    input_buffer: InputBuffer,
    output_buffer: OutputBuffer,
    aframe: Architecture,
    data_it: Iterable[Tuple[torch.Tensor, float, bool]],
    update_size: float,
    time_offset: float,
    device: str,
):
    integrated = None

    # flag that declares if the most previous frame
    # was analysis ready or not
    in_spec = False

    state = snapshotter.initial_state
    for X, t0, ready in data_it:
        # if this frame was not analysis ready
        if not ready:
            if searcher.detecting:
                # if we were in the middle of a detection,
                # we won't get to see the peak of the event
                # so build the event with what we have
                event = searcher.build_event(
                    integrated[-1], t0 - update_size, len(integrated) - 1
                )
                if event is not None:
                    # maybe process event found in the previous frame
                    # event_processor(event)
                    event_queue.put((event, input_buffer))
                    searcher.detecting = False

            # check if this is because the frame stream stopped
            # being analysis ready, in which case perform updates
            # but don't search for events
            if X is not None:
                logging.warning(
                    "Frame {} is not analysis ready. Performing "
                    "inference but ignoring any triggers".format(t0)
                )
            # or if it's because frames were dropped within the stream
            # in which case we should reset our states
            else:
                logging.warning(
                    "Missing frame files after timestep {}, "
                    "resetting states".format(t0)
                )

                input_buffer.reset()
                output_buffer.reset()

                # nothing left to do, so move on to next frame
                continue

        elif not in_spec:
            # the frame is analysis ready, but previous frames
            # weren't, so reset our running states
            logging.info(f"Frame {t0} is ready again, resetting states")
            state = snapshotter.reset()
            input_buffer.reset()
            output_buffer.reset()
            in_spec = True

        # we have a frame that is analysis ready,
        # so lets analyze it:
        X = X.to(device)

        # update the snapshotter state and return
        # unfolded batch of overlapping windows
        batch, state = snapshotter(X[None], state)

        # whiten the batch, and analyze with aframe
        whitened = whitener(batch)

        y = aframe(whitened)[:, 0]

        # update our input buffer with latest strain data,
        input_buffer.update(X.cpu(), t0)
        # update our output buffer with the latest aframe output,
        # which will also automatically integrate the output
        integrated = output_buffer.update(y.cpu(), t0)

        # if this frame was analysis ready,
        # and we had enough previous to build whitening filter
        # search for events in the integrated output
        event = None
        if snapshotter.full_psd_present:  # and ready:
            event = searcher.search(integrated, t0 + time_offset)

        # if we found an event, process it!
        if event is not None:
            # event_processor(event)
            logging.info("Putting event in event queue")
            event_queue.put(
                (
                    event,
                    input_buffer,
                )
            )
            searcher.detecting = False
        # TODO write buffers to disk:


def pastro_subprocess(
    pastro_queue: Queue,
    server: GdbServer,
    background_path: Path,
    foreground_path: Path,
    rejected_path: Path,
    astro_event_rate: float,
    outdir: Path,
):
    gdb = gracedb_factory(server, outdir)

    logging.info("Fitting p_astro model or loading from cache")
    pastro_model = fit_or_load_pastro(
        outdir / "pastro.pkl",
        background_path,
        foreground_path,
        rejected_path,
        astro_event_rate=astro_event_rate,
    )
    logging.info("Loaded p_astro model")

    while True:
        event = pastro_queue.get()
        logging.info("Calculating p_astro")
        pastro = pastro_model(event.detection_statistic)
        logging.info(f"Submitting p_astro: {pastro}")
        graceid = pastro_queue.get()
        gdb.submit_pastro(float(pastro), graceid, event.gpstime)


def amplfi_subprocess(
    amplfi_queue: Queue,
    server: GdbServer,
    outdir: Path,
    amplfi_weights: Path,
    inference_params: List[str],
    sample_rate: float,
    fduration: float,
    fftlength: Optional[float] = None,
    highpass: Optional[float] = None,
    samples_per_event: int = 20000,
    nside: int = 32,
    device: str = "cpu",
):
    gdb = gracedb_factory(server, outdir)

    norm_layer = GroupNorm1DGetter(groups=8)
    embedding_net = MultiModalPsd(
        num_ifos=2,
        time_context_dim=12,
        freq_context_dim=32,
        time_layers=[5, 3, 3],
        freq_layers=[5, 3, 3],
        norm_layer=norm_layer,
    )
    amplfi_architecture = MaskedAutoregressiveFlow(
        num_params=len(inference_params),
        transform_type="spline",
        hidden_features=100,
        num_transforms=60,
        num_blocks=6,
        embedding_net=embedding_net,
    )

    logging.info(f"Loading AMPLFI from weights at path {amplfi_weights}")
    amplfi, scaler = load_amplfi(
        amplfi_architecture, amplfi_weights, len(inference_params)
    )
    amplfi = amplfi.to(device)
    scaler = scaler.to(device)

    spectral_density = SpectralDensity(
        sample_rate=sample_rate,
        fftlength=fftlength,
        average="median",
    ).to(device)
    amplfi_whitener = Whiten(
        fduration=fduration, sample_rate=sample_rate, highpass=highpass
    ).to(device)

    while True:
        event, buffer = amplfi_queue.get()
        posterior, skymap, figure = run_amplfi(
            event_time=event.gpstime,
            input_buffer=buffer,
            inference_params=inference_params,
            samples_per_event=samples_per_event,
            spectral_density=spectral_density,
            amplfi_whitener=amplfi_whitener,
            amplfi=amplfi,
            std_scaler=scaler,
            nside=nside,
            device=device,
        )
        graceid = amplfi_queue.get()
        gdb.submit_pe(posterior, figure, skymap, graceid, event.gpstime)


def event_creation_subprocess(
    event_queue: Queue,
    server: GdbServer,
    outdir: Path,
    amplfi_queue: Queue,
    pastro_queue: Queue,
):
    gdb = gracedb_factory(server, outdir)
    while True:
        event, buffer = event_queue.get()
        logging.info("Putting event in amplfi and pastro queues")
        amplfi_queue.put((event, buffer))
        pastro_queue.put(event)

        # write event information to disk
        # and submit it to gracedb
        event.write(outdir)
        response = gdb.submit(event)

        # Get the event's graceid for submitting
        # further data products
        if isinstance(response, str):
            # If the response is a string, then we are using the local gracedb
            # client, which just returns the filename
            graceid = response
        else:
            graceid = response.json()["graceid"]
        logging.info("Putting graceid in amplfi and pastro queues")
        amplfi_queue.put(graceid)
        pastro_queue.put(graceid)


def main(
    aframe_weights: Path,
    amplfi_architecture: FlowArchitecture,
    amplfi_weights: Path,
    background_path: Path,
    foreground_path: Path,
    rejected_path: Path,
    outdir: Path,
    datadir: Path,
    ifos: List[str],
    inference_params: List[str],
    channels: List[str],
    sample_rate: float,
    kernel_length: float,
    inference_sampling_rate: float,
    psd_length: float,
    aframe_right_pad: float,
    amplfi_kernel_length: float,
    event_position: float,
    fduration: float,
    integration_window_length: float,
    astro_event_rate: float,
    data_source: str = "frames",
    fftlength: Optional[float] = None,
    highpass: Optional[float] = None,
    lowpass: Optional[float] = None,
    refractory_period: float = 8,
    far_threshold: float = 1,
    server: GdbServer = "test",
    ifo_suffix: str = None,
    input_buffer_length: int = 75,
    output_buffer_length: int = 8,
    samples_per_event: int = 20000,
    nside: int = 32,
    device: str = "cpu",
):
    """
    Main function for launching real-time Aframe and AMPLFI pipeline.

    Args:
        aframe_weights:
            Path to trained Aframe model weights
        amplfi_architecture:
            AMPLFI model architecture for parameter estimation
        amplfi_weights:
            Path to trained AMPLFI model weights
        background_path:
            Path to background noise events dataset
            used for Aframe FAR calculation
        foreground_path:
            Path to recovered injection events dataset
        rejected_path:
            Path to rejected injection parameters dataset
        outdir:
            Directory to save output files
        datadir:
            Directory containing input strain data
        ifos:
            List of interferometer names
        inference_params:
            List of parameters on which the AMPLFI
            model was trained to perform inference
        channels:
            List of the channel names to analyze
        sample_rate:
            Input data sample rate in Hz
        kernel_length:
            Length of Aframe analysis kernel in seconds
        inference_sampling_rate:
            Rate at which to sample the output of the Aframe model
        psd_length:
            Length of PSD estimation window in seconds
        aframe_right_pad:
            Time offset for trigger positioning in seconds
        amplfi_kernel_length:
            Length of AMPLFI analysis window in seconds
        event_position:
            Event position (in seconds) from the left edge
            of the analysis window used for parameter estimation
        fduration:
            Length of whitening filter in seconds
        integration_window_length:
            Length of output integration window in seconds
        astro_event_rate:
            Prior on rate of astrophysical events in units Gpc^-3 yr^-1
        fftlength:
            FFT length in seconds (defaults to kernel_length + fduration)
        highpass:
            High-pass filter frequency in Hz
        lowpass:
            Low-pass filter frequency in Hz
        refractory_period:
            Minimum time between events in seconds
        far_threshold:
            False alarm rate threshold in events/day
        server:
            GraceDB server to use:
            "local", "playground", "test" or "production"
        ifo_suffix:
            Optional suffix for accessing data from /dev/shm.
            Useful when analyzing alternative streams like
            MDC replays that have directory structures in the format
            `{ifo}_{ifo_suffix}`
        input_buffer_length:
            Length of strain data buffer in seconds
        output_buffer_length:
             Length of inference output buffer in seconds
        samples_per_event:
            Number of posterior samples to generate per event
            for creating skymaps and other parameter estimation
            data products
        nside:
            Healpix resolution for low-latency skymaps
        device:
            Device to run inference on ("cpu" or "cuda")
    """
    # run htgettoken and kinit
    if server != "local":
        authenticate()

    fftlength = fftlength or kernel_length + fduration

    mp.set_start_method("spawn")
    amplfi_queue = Queue()
    args = (
        amplfi_queue,
        server,
        outdir,
        amplfi_weights,
        inference_params,
        sample_rate,
        fduration,
        fftlength,
        highpass,
        samples_per_event,
        nside,
        device,
    )
    amplfi_process = Process(target=amplfi_subprocess, args=args)
    amplfi_process.start()

    pastro_queue = Queue()
    args = (
        pastro_queue,
        server,
        background_path,
        foreground_path,
        rejected_path,
        astro_event_rate,
        outdir,
    )
    pastro_process = Process(target=pastro_subprocess, args=args)
    pastro_process.start()

    event_queue = Queue()
    args = (event_queue, server, outdir, amplfi_queue, pastro_queue)
    event_process = Process(target=event_creation_subprocess, args=args)
    event_process.start()

    if data_source == "ngdd":
        update_size = 1 / 16
        data_it = ngdd_data_iterator(
            strain_channels=channels,
            ifos=ifos,
            sample_rate=sample_rate,
        )
    elif data_source == "frames":
        update_size = 1
        data_it = data_iterator(
            datadir=datadir,
            channels=channels,
            ifos=ifos,
            sample_rate=sample_rate,
            ifo_suffix=ifo_suffix,
            timeout=10,
        )
    else:
        raise ValueError(
            f"Invalid data source {data_source}. Must be 'ngdd' or 'frames'"
        )

    # initialize a buffer for storing recent strain data,
    # and for storing integrated aframe outputs
    input_buffer = InputBuffer(
        ifos=ifos,
        sample_rate=sample_rate,
        buffer_length=input_buffer_length,
        fduration=fduration,
        amplfi_kernel_length=amplfi_kernel_length,
        event_position=event_position,
        device="cpu",
    )

    output_buffer = OutputBuffer(
        inference_sampling_rate=inference_sampling_rate,
        integration_window_length=integration_window_length,
        buffer_length=output_buffer_length,
        device="cpu",
    )

    # Load in Aframe and amplfi models
    logging.info(f"Loading Aframe from weights at path {aframe_weights}")
    aframe = torch.jit.load(aframe_weights)
    aframe = aframe.to(device)

    whitener = BatchWhitener(
        kernel_length=kernel_length,
        sample_rate=sample_rate,
        inference_sampling_rate=inference_sampling_rate,
        batch_size=update_size * inference_sampling_rate,
        fduration=fduration,
        fftlength=fftlength,
        highpass=highpass,
        lowpass=lowpass,
    ).to(device)

    snapshotter = OnlineSnapshotter(
        update_size=update_size,
        num_channels=len(ifos),
        psd_length=psd_length,
        kernel_length=kernel_length,
        fduration=fduration,
        sample_rate=sample_rate,
        inference_sampling_rate=inference_sampling_rate,
    ).to(device)

    # Amplfi setup
    spectral_density = SpectralDensity(
        sample_rate=sample_rate,
        fftlength=fftlength,
        average="median",
    ).to(device)
    pe_whitener = Whiten(
        fduration=fduration,
        sample_rate=sample_rate,
        highpass=highpass,
        lowpass=lowpass,
    ).to(device)

    # load in background, foreground, and rejected injections;
    # use them to fit (or load in a cached) a pastro model
    logging.info("Loading background distribution")
    background = EventSet.read(background_path)

    # if event set is not sorted by detection statistic, sort it
    # which will significantly speed up the far calculation
    if not background.is_sorted_by("detection_statistic"):
        logging.info(
            "Sorting background by detection statistic and writing to disk"
        )
        background = background.sort_by("detection_statistic")
        background.write(background_path)

    # Convert FAR to Hz from 1/days
    far_threshold /= SECONDS_PER_DAY
    searcher = Searcher(
        background=background,
        far_threshold=far_threshold,
        inference_sampling_rate=inference_sampling_rate,
        refractory_period=refractory_period,
        ifos=ifos,
        channels=channels,
        datadir=datadir,
        ifo_suffix=ifo_suffix,
    )

    time_offset = get_time_offset(
        inference_sampling_rate,
        fduration,
        integration_window_length,
        kernel_length,
        aframe_right_pad,
    )

    logging.info("Beginning search")
    search(
        gdb=gdb,
        buffer=input_buffer,
        spectral_density=spectral_density,
        pe_whitener=pe_whitener,
        amplfi=amplfi,
        scaler=scaler,
        pastro_model=pastro_model,
        data_it=data_it,
        update_size=update_size,
        time_offset=time_offset,
        samples_per_event=samples_per_event,
        inference_params=inference_params,
        outdir=outdir,
        nside=nside,
        device=device,
    )


if __name__ == "__main__":
    main()
