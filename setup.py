#!/usr/bin/env python3
"""
setup.py — CORDIS
=================
Packaging for CORDIS: a scalable coordinated resource-allocation framework for
distributed cell-free ISAC (decentralized beamforming + power allocation via
consensus ADMM, with a centralized ceiling and classical baselines).

Installing the package (``pip install -e .``) makes ``cordis`` importable from
any working directory, which is what the experiment/plot launchers and the
validation suite expect.

    python -m venv .venv && source .venv/bin/activate
    pip install --upgrade pip
    pip install -e .                 # core install
    pip install -e ".[notebooks]"    # + Jupyter playgrounds
    pip install -e ".[assent]"       # + learning-based sensing association
"""
from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent


def _read(rel: str, default: str = "") -> str:
    p = ROOT / rel
    return p.read_text(encoding="utf-8") if p.exists() else default


def _requirements() -> list[str]:
    """Parse runtime deps from requirements.txt (ignoring comments/blanks)."""
    reqs: list[str] = []
    for line in _read("requirements.txt").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and not line.startswith("-"):
            reqs.append(line)
    return reqs


setup(
    name="cordis",
    version="0.1.0",
    description=(
        "Coordinated Resource allocation for Distributed ISAC Systems: "
        "scalable decentralized beamforming and power allocation for "
        "cell-free ISAC via consensus ADMM."
    ),
    long_description=_read("README.md"),
    long_description_content_type="text/markdown",
    author="Mehdi Zafari",
    url="https://github.com/LS-Wireless/CORDIS",
    project_urls={
        "Source": "https://github.com/LS-Wireless/CORDIS",
        "Issues": "https://github.com/LS-Wireless/CORDIS/issues",
    },
    license="MIT",
    packages=find_packages(include=["cordis", "cordis.*"]),
    python_requires=">=3.10",
    install_requires=_requirements() or [
        "numpy>=1.26,<3.0",
        "scipy>=1.11",
        "cvxpy>=1.5",
        "joblib>=1.3",
        "matplotlib>=3.7",
        "tqdm>=4.65",
    ],
    extras_require={
        # Interactive plot playgrounds under notebooks/.
        "notebooks": ["jupyterlab>=4.0", "ipykernel"],
        # Learning-based sensing AP-target association (optional integration).
        "assent": [
            "assent @ git+https://github.com/LS-Wireless/ASSENT-CellFree-ISAC",
        ],
        # Test/validation tooling.
        "dev": ["pytest>=7.0"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
    ],
    include_package_data=True,
    zip_safe=False,
)

