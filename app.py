# Create shiny app that uses bayesian updating rule that assumes normal distributions

import scipy.stats as scistats
import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl

mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False

from shiny import render, reactive
from shiny.express import input
from shiny.express import ui  # as ui

with ui.layout_columns():
    ui.input_numeric("mu_pop", "Population mean", 15, step=0.1)

    ui.input_numeric(
        "sig_pop", "Population StdDev (between person variation)", 1, min=0, step=0.1
    )
    ui.input_numeric(
        "sig_bio", "Bio StdDev (in-person variation)", 0.4, min=0, step=0.1
    )
    ui.input_numeric(
        "sig_meas", "Measurement StdDev (measurement variation)", 0.7, min=0, step=0.1
    )

    ui.input_numeric("y1", "First measurement", 13, step=0.1)
    ui.input_numeric("y2", "Second measurement", None, step=0.1)
    ui.input_numeric("cut", "Threshold for above", 13, step=0.1)

    ui.input_checkbox_group(
        "post_type",
        "Posterior type",
        {"inst": "Instantaneous posterior", "ss": "Steady state posterior"},
        selected="inst",
    )

    p_above_inst = reactive.value(None)
    p_above_ss = reactive.value(None)


def bayesian_update_normal(prior_mean, sample_mean, prior_sig, sample_sig):
    """Bayesian update for normal distributions."""
    posterior_mean = (prior_mean / prior_sig**2 + sample_mean / sample_sig**2) / (
        1 / prior_sig**2 + 1 / sample_sig**2
    )
    posterior_sig = np.sqrt(1 / (1 / prior_sig**2 + 1 / sample_sig**2))
    return posterior_mean, posterior_sig


@render.plot
def plot():
    mu_pop = input.mu_pop()
    sig_pop = input.sig_pop()
    sig_bio = input.sig_bio()
    sig_meas = input.sig_meas()
    y1 = input.y1()
    y2 = input.y2()
    cut = input.cut()
    
    sig_pop_tot = (sig_pop**2 + sig_bio**2) ** 0.5
    sig_meas_tot = (sig_meas**2 + sig_bio**2) ** 0.5

    
    #Set min, max x-axis
    min_x = np.min(
        [
            mu_pop - 4 * sig_pop_tot,
            y1 - 3 * sig_meas_tot,
            y2 - 3 * sig_meas_tot if y2 is not None else np.inf,
        ]
    )
    max_x = np.max(
        [
            mu_pop + 4 * sig_pop_tot,
            y1 + 3 * sig_meas_tot,
            y2 - 3 * sig_meas_tot if y2 is not None else -np.inf,
        ]
    )

    x = np.linspace(min_x, max_x, 1000)

    #Use update formula
    mu_inst_post, sig_inst_post = bayesian_update_normal(
        mu_pop, y1, sig_pop, sig_meas_tot
    )
    mu_ss_post, sig_ss_post = bayesian_update_normal(
        mu_pop, y1, sig_pop_tot, sig_meas
    )
    alpha_meas = 0.5
    alpha_pop = 0.6
    
    plt.figure(figsize=(12, 6))
    if "inst" in input.post_type():
        plt.plot(
            x,
            scistats.norm.pdf(x, mu_pop, sig_pop),
            label="Population distribution",
            color="C0",
            ls="--",
            alpha=alpha_pop,
        )
        plt.plot(
            x,
            scistats.norm.pdf(x, y1, sig_meas_tot),
            label="1st Measurement+bio distribution",
            color="C0",
            ls=":",
            alpha=alpha_meas,
        )
        plt.plot(
            x,
            scistats.norm.pdf(x, mu_ss_post, sig_ss_post),
            label="Steady state posterior",
            color="C0",
            lw=2,
        )
    if "ss" in input.post_type():
        plt.plot(
            x,
            scistats.norm.pdf(x, y1, sig_meas),
            label="1st Measurement distribution",
            color="C1",
            ls=":",
            alpha=alpha_meas,
        )
        plt.plot(
            x,
            scistats.norm.pdf(x, mu_pop, sig_pop_tot),
            label="Population+bio distribution",
            color="C1",
            ls="--",
            alpha=alpha_pop,
        )
        plt.plot(
            x,
            scistats.norm.pdf(x, mu_inst_post, sig_inst_post),
            label="Instantaneous posterior",
            color="C1",
            lw=2,
        )

    if y2 is not None:
        #Second measurement update
        mu_inst_post, sig_inst_post = bayesian_update_normal(
            mu_inst_post, y2, sig_inst_post, sig_meas_tot
        )
        mu_ss_post, sig_ss_post = bayesian_update_normal(
            mu_ss_post, y2, sig_ss_post, sig_meas
        )

    plt.plot(y1, 0, marker="o", color="k", label="First measurement", ls="")

    if y2 is not None:
        if "inst" in input.post_type():
            plt.plot(
                x,
                scistats.norm.pdf(x, y2, sig_meas),
                label="2nd Measurement distribution",
                color="C0",
                ls="-.",
                alpha=alpha_meas,
            )
        if "ss" in input.post_type():
            plt.plot(
                x,
                scistats.norm.pdf(x, y2, sig_meas_tot),
                label="2nd Measurement+bio distribution",
                color="C1",
                ls="-.",
                alpha=alpha_meas,
            )

        plt.plot(y2, 0, marker="o", color="grey", label="First measurement", ls="")

    plt.legend()
    #Probability of being above the cut (no rounding!)
    p_above_ss.set(scistats.norm.sf(cut, mu_ss_post, sig_ss_post))
    p_above_inst.set(scistats.norm.sf(cut, mu_inst_post, sig_inst_post))


@render.ui
def _():
    """Render the UI for the posterior probabilities."""
    return ui.div(
        ui.h3("Posterior probabilities"),
        ui.p(f"Probability of being above {input.cut()}:"),
        ui.p(f"Instantaneous: {p_above_inst.get():.3f}"),
        ui.p(f"Steady state: {p_above_ss.get():.3f}"),
    )
