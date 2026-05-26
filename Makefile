# =============================================================================
#  CORDIS — Makefile
# =============================================================================
#
#  Reproducible paper-figure pipeline.  Common usage:
#
#      make help               # print this list
#      make sinr_cdf           # run one experiment
#      make plot-sinr_cdf      # plot the latest run for that experiment
#      make all                # run every experiment
#      make figures            # plot every figure (from latest)
#      make validate           # run all stage validators (from 8a/b/c/d to end)
#      make submit-sinr_cdf    # sbatch the SLURM wrapper for one experiment
#      make submit-all         # sbatch every experiment
#      make clean-figures      # rm -rf figures/   (DESTRUCTIVE)
#
#  The EXPERIMENTS list below is the single source of truth and must
#  match cordis.experiments.REGISTRY (validate_stage8d.py enforces this).
#
# =============================================================================

# ─── Tunables ────────────────────────────────────────────────────────────────
PYTHON ?= python3
SBATCH ?= sbatch

# Make `cordis` importable for every recipe (no `pip install -e .` required).
export PYTHONPATH := $(CURDIR):$(PYTHONPATH)

# ─── Experiments ─────────────────────────────────────────────────────────────
EXPERIMENTS := \
    sinr_cdf scnr_cdf \
    gamma_sweep kappa_sweep clutter_cnr_sweep snr_sweep \
    n_ue_sweep n_ap_sweep antennas_sweep \
    convergence_trace fronthaul_table

# Derived target sets.
PLOT_TARGETS   := $(addprefix plot-,   $(EXPERIMENTS))
SUBMIT_TARGETS := $(addprefix submit-, $(EXPERIMENTS))

.PHONY: help test validate \
        all figures \
        submit-all \
        clean-figures clean-results clean-logs \
        regenerate-scripts \
        $(EXPERIMENTS) $(PLOT_TARGETS) $(SUBMIT_TARGETS)


# ─── Help ────────────────────────────────────────────────────────────────────
help:
	@echo "CORDIS — paper-figure pipeline"
	@echo
	@echo "Run experiments:"
	@echo "  make <name>             where <name> is one of:"
	@printf "                            "
	@echo $(EXPERIMENTS) | tr ' ' '\n' | awk '{printf "%-22s", $$0; if (NR%2==0) printf "\n                            "} END {print ""}'
	@echo "  make all                run every experiment in sequence"
	@echo
	@echo "Plot figures (from latest run of each experiment):"
	@echo "  make plot-<name>        plot one figure"
	@echo "  make figures            plot every figure"
	@echo
	@echo "Submit to SLURM (one job per experiment):"
	@echo "  make submit-<name>      sbatch scripts/slurm/exp_<name>.sbatch"
	@echo "  make submit-all         submit every experiment"
	@echo
	@echo "Validate:"
	@echo "  make validate           run all stage validators (from 8a/b/c/d to end)"
	@echo "  make test               run pytest"
	@echo
	@echo "Maintenance:"
	@echo "  make regenerate-scripts python3 scripts/regenerate_experiment_scripts.py"
	@echo "  make clean-figures      rm -rf figures/"
	@echo "  make clean-logs         rm -rf logs/"
	@echo "  make clean-results      rm -rf results/   (DESTRUCTIVE — asks first)"


# ─── Per-experiment run targets ──────────────────────────────────────────────
#
#  `make sinr_cdf` runs the journal-paper-sized configuration via the
#  scripts/exp_sinr_cdf.sh launcher.  Override knobs from the command
#  line, e.g.   make sinr_cdf N_DROPS=10 N_WORKERS=4
#
$(EXPERIMENTS):
	bash scripts/exp_$@.sh


# ─── Per-experiment plot targets ─────────────────────────────────────────────
#
#  `make plot-sinr_cdf` plots the most recent results/exp_sinr_cdf/<ts>/.
#  Override with    make plot-sinr_cdf EXP_DIR=results/exp_sinr_cdf/<ts>
#
EXP_DIR ?=
$(PLOT_TARGETS): plot-%:
	@if [ -n "$(EXP_DIR)" ]; then \
	    $(PYTHON) scripts/plot_$*.py --exp-dir "$(EXP_DIR)" ; \
	else \
	    $(PYTHON) scripts/plot_$*.py ; \
	fi


# ─── Per-experiment SLURM submit targets ─────────────────────────────────────
$(SUBMIT_TARGETS): submit-%:
	@mkdir -p logs/slurm
	$(SBATCH) scripts/slurm/exp_$*.sbatch


# ─── Aggregate targets ───────────────────────────────────────────────────────
all: $(EXPERIMENTS)

figures: $(PLOT_TARGETS)

submit-all: $(SUBMIT_TARGETS)


# ─── Validation ──────────────────────────────────────────────────────────────
validate:
	$(PYTHON) scripts/validate_stage8.py

test:
	$(PYTHON) -m pytest tests/


# ─── Maintenance ─────────────────────────────────────────────────────────────
regenerate-scripts:
	$(PYTHON) scripts/regenerate_experiment_scripts.py
	$(PYTHON) notebooks/_build_playgrounds.py

clean-figures:
	rm -rf figures/

clean-logs:
	rm -rf logs/

clean-results:
	@echo
	@echo "  ⚠  This will permanently delete EVERYTHING in results/."
	@echo "     Press Ctrl-C within 5 seconds to abort."
	@echo
	@sleep 5
	rm -rf results/

