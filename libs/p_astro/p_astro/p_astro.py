from typing import TypeVar

import astropy
import numpy as np
from astropy.cosmology import Planck15
from ledger.events import EventSet, RecoveredInjectionSet
from ledger.injections import InjectionParameterSet
from numpy.polynomial import Polynomial
from scipy.integrate import quad
from scipy.stats import gaussian_kde

F = TypeVar("F", float, np.ndarray)


def calculate_injected_volume(
    rejected: InjectionParameterSet,
    recovered: RecoveredInjectionSet,
    cosmology: astropy.Cosmology = Planck15,
):
    def volume_element(z: F) -> F:
        return cosmology.differential_comoving_volume(z).value / (1 + z)

    redshifts = np.concatenate([rejected.redshift, recovered.redshift])
    decs = np.concatenate([rejected.dec, recovered.dec])
    zmin, zmax = redshifts.min(), redshifts.max()
    decmin, decmax = decs.min(), decs.max()

    volume, _ = quad(lambda z: volume_element(z), zmin, zmax)
    theta_max = np.pi / 2 - decmin
    theta_min = np.pi / 2 - decmax
    omega = -2 * np.pi * (np.cos(theta_max) - np.cos(theta_min))
    return volume * omega


def fit_background(
    background: EventSet,
) -> callable[F, F]:
    kde = gaussian_kde(background.detection_statistic)
    min_ = np.min(background.detection_statistic)
    max_ = np.max(background.detection_statistic)
    # Estimate the peak of the distribution
    samples = np.linspace(
        min_,
        max_,
        1000,
    )
    pdf = kde(samples)

    # Determine the range of values to use for fitting
    # a line to a portion of the pdf.
    # Roughly, we have too few samples to properly
    # estimate the KDE once the pdf drops below 1/sqrt(N)
    peak_idx = np.argmax(pdf)
    threshold_pdf_value = 1 / np.sqrt(len(background))
    start = np.argmin(pdf[peak_idx:] > 10 * threshold_pdf_value) + peak_idx
    stop = np.argmin(pdf[peak_idx:] > threshold_pdf_value) + peak_idx

    # Fit a line to the log pdf of the region
    fit_samples = samples[start:stop]
    tail = Polynomial.fit(fit_samples, np.log(pdf[start:stop]), 1)
    threshold_statistic = samples[start]

    def model(statistics: F) -> F:
        if statistics < threshold_statistic:
            density = kde(statistics)
        else:
            density = np.exp(tail(statistics))
        return density * len(background) / background.Tb

    return model


def fit_foreground(
    foreground: RecoveredInjectionSet,
    rejected_parameters: InjectionParameterSet,
    astro_event_rate: float,
    cosmology: astropy.Cosmology,
):
    injected_volume = calculate_injected_volume(
        rejected_parameters, foreground, cosmology
    )

    percent_accepted = len(foreground) / (
        len(rejected_parameters) + len(foreground)
    )
    scaling_factor = astro_event_rate * percent_accepted * injected_volume
    kde = gaussian_kde(foreground.detection_statistic)

    def model(statistics: F) -> F:
        return kde(statistics) * scaling_factor

    return model


def fit_p_astro(
    background: EventSet,
    foreground: RecoveredInjectionSet,
    rejected_parameters: InjectionParameterSet,
    astro_event_rate: float,
    cosmology: astropy.Cosmology,
):
    background = fit_background(background)
    foreground = fit_foreground(
        foreground, rejected_parameters, astro_event_rate, cosmology
    )

    def p_astro(statistics: F) -> F:
        return foreground(statistics) / (
            foreground(statistics) + background(statistics)
        )

    return p_astro
