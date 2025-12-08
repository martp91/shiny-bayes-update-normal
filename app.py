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


#Add title and explanatory text
ui.h2("Bayesian updating of Hb measurements"),
ui.p(
    "This app can be used to determine the probability that a Hb measurement"
    "is below the threshold for donation. "
    "You can input one or two measurements, and the app will calculate "
    "the posterior distribution of the true value using Bayesian updating rules."
),
ui.p(
    "The model accounts for population variability, biological variability, "
    "and measurement variability."
),
ui.hr(),

with ui.layout_columns():
    ui.input_numeric("y1", "First measurement", 7.8, step=0.1)
    ui.input_numeric("y2", "Second measurement", None, step=0.1)
    ui.input_radio_buttons("sex", "Sex", {'1': 'Female', '2': 'Male'}, selected='1')
      
    #Hide these inputs in the UI but keep them for calculations
    p_above_inst = reactive.value(None)
    p_above_ss = reactive.value(None)
    @reactive.effect 
    def set_default():
        if input.sex() == '1':
            ui.update_numeric('cut', value=7.8)
            ui.update_numeric('mu_pop', value=8.5)
        else:
            ui.update_numeric('cut', value=8.4)
            ui.update_numeric('mu_pop', value=9.4)
    
    ui.input_action_button("toggle_button", "Show/Hide Extra inputs")

with ui.panel_conditional(
    "input.toggle_button % 2 == 1"):
    with ui.layout_columns():
        ui.input_numeric("cut", "Threshold for above", 7.8, step=0.1)
        ui.input_numeric("mu_pop", "Population mean", 8.5, step=0.1)
        ui.input_numeric(
            "sig_pop", "Population StdDev (between person variation)", 0.55, min=0, step=0.1
        )
        ui.input_numeric(
            "sig_bio", "Bio StdDev (in-person variation)", 0.2, min=0, step=0.1
        )
        ui.input_numeric(
            "sig_meas", "Measurement StdDev (measurement variation)", 0.38, min=0, step=0.1
        )
        ui.input_checkbox_group(
            "post_type",
            "Posterior type",
            {"inst": "Instantaneous posterior", "ss": "Steady state posterior"},
            selected="inst",
        )
    
    


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
    plt.axvline(cut, color="r", ls=":", label="Threshold")
    if "inst" in input.post_type():
        plt.plot(
            x,
            scistats.norm.pdf(x, mu_pop, sig_pop),
            label="Population distribution",
            color="C0",
            ls="--",
            alpha=alpha_pop,
        )
        if y1 is not None and y2 is None:
            # plt.plot(
            #     x,
            #     scistats.norm.pdf(x, y1, sig_meas_tot),
            #     label="1st Measurement+bio distribution",
            #     color="C0",
            #     ls=":",
            #     alpha=alpha_meas,
            # )
            plt.plot(
                x,
                scistats.norm.pdf(x, mu_ss_post, sig_ss_post),
                label="Steady state posterior",
                color="C0",
                lw=2,
            )
    if "ss" in input.post_type():
        if y1 is not None and y2 is None:
            # plt.plot(
            #     x,
            #     scistats.norm.pdf(x, y1, sig_meas),
            #     label="1st Measurement distribution",
            #     color="C1",
            #     ls=":",
            #     alpha=alpha_meas,
            # )
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
            #TODO: show these with toggle
            # plt.plot(
            #     x,
            #     scistats.norm.pdf(x, y2, sig_meas),
            #     label="2nd Measurement distribution",
            #     color="C0",
            #     ls="-.",
            #     alpha=alpha_meas,
            # )
            plt.plot(
                x,
                scistats.norm.pdf(x, mu_ss_post, sig_ss_post),
                label="Steady state posterior",
                color="C0",
                lw=2,
            )
        if "ss" in input.post_type():
            # plt.plot(
            #     x,
            #     scistats.norm.pdf(x, y2, sig_meas_tot),
            #     label="2nd Measurement+bio distribution",
            #     color="C1",
            #     ls="-.",
            #     alpha=alpha_meas,
            # )
            plt.plot(
                x,
                scistats.norm.pdf(x, mu_inst_post, sig_inst_post),
                label="Instantaneous posterior",
                color="C1",
                lw=2,
            )

        plt.plot(y2, 0, marker="o", color="grey", label="Second measurement", ls="")
    plt.xlabel('Hb [mmol/L]')
    plt.ylabel('Prob. density')
    plt.legend()
    #Probability of being above the cut (no rounding!)
    p_above_ss.set(scistats.norm.sf(cut, mu_ss_post, sig_ss_post))
    p_above_inst.set(scistats.norm.sf(cut, mu_inst_post, sig_inst_post))


@render.ui
def _():
    """Render the UI for the posterior probabilities."""
    if 'inst' in input.post_type():
        p = p_above_inst.get()
    else:
        p = p_above_ss.get()
    
    return ui.div(
        ui.h3("Posterior probabilities"),
        ui.p(f"Probability of being above {input.cut()}:"),
        # ui.p(f"{p_above_inst.get():.4f}") if input.post_type() == 'inst' else ui.p(f"{p_above_ss.get():.4f}"),
        ui.p(f"{p:.5f}"),
        # ui.p(f"Steady state: {p_above_ss.get():.3f}" if input.post_type() == 'ss' else ""),
    )
