<div align="center">
        <picture>
                <source media="(prefers-color-scheme: dark)" srcset="./docs/assets/logo.png">
                <source media="(prefers-color-scheme: light)" srcset="./docs/assets/logo.png">
                <img alt="Library Banner" src="./docs/assets/logo.png" width="500" height="500">
        </picture>
</div>

<br>

<div align="center">
  <a href="#">
        <img src="https://img.shields.io/badge/Python-%E2%89%A53.9-efefef">
    </a>
    <a href="https://github.com/Pruneeuh/POSEIDON/actions/workflows/python-tests.yml">
        <img alt="Tox" src="https://github.com/Pruneeuh/POSEIDON/actions/workflows/python-tests.yml/badge.svg">
    </a>
    <a href="https://github.com/Pruneeuh/POSEIDON/actions/workflows/python-linter.yml">
        <img alt="Lint" src="https://github.com/Pruneeuh/POSEIDON/actions/workflows/python-linter.yml/badge.svg">
    </a>
    <a href="https://pepy.tech/project/poseidon">
        <img alt="Pepy" src="https://static.pepy.tech/badge/poseidon_torch">
    </a>
    <a href="#">
        <img src="https://img.shields.io/badge/License-MIT-efefef">
    </a>
    <br>
    <a href="https://Pruneeuh.github.io/POSEIDON/"><strong>Explore POSEIDON docs »</strong></a>
</div>
<br>

# POSEIDON
POSe estimation with Explicit / Implicit Differentiable OptimisatioN

**POSEIDON** is a Python project focused on estimating the 3D position (pose) of a camera using the P3P (Perspective-3-Point) algorithm. It offers a clean and modular implementation of the core algorithm for precise geometric computations.

## 📚 Table of Contents
- [Motivation](#motivation)
- [Usage](#usage)
- [Tutorials](#-tutorials)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Tests](#tests)
- [Installation](#-installation)
- [Contributing](#-contributing)
- [Acknowledgments](#-acknowledgments)

# Motivation
**Optimizing the YOLO-NAS Learning Function for Vision-Based Landing via Differentiable PnP Integration Internship Context**
Vision-Based Landing (VBL) is a complementary approach to traditional landing systems such as ILS (Instrument Landing System) and GPS. It uses visual information and artificial intelligence models to estimate the 3D position of an aircraft (latitude, longitude, altitude) during landing.
A derivative of the YOLO object detection framework, YOLO-NAS, is explored for this task. The model is used to detect the runway and localize its four corners, whose 3D coordinates are known. These detections are then used in a PnP (Perspective-n-Point) algorithm to estimate the aircraft’s pose.
However, directly applying a joint estimation learning objective—originally designed for YOLO-NAS—does not align well with industrial requirements. In practice, vision-based systems are expected to meet accuracy tolerances that depend on the aircraft's distance to the runway, as defined by systems like ILS. A VBL system that respects the same tolerances is considered acceptable.
These error margins can potentially be incorporated directly into the training phase by integrating the pose estimation process (PnP) into the model’s loss function.

# Usage
**A compléter**

## 🔥 Tutorials

| **Tutorial Name**           | Notebook                                                                                                                                                           |
| :-------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Apply the P3P algorithm | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pruneeuh/POSEIDON/blob/main/tutorials/Tutorial_1_P3P.ipynb)            |
| Apply the P3P algorithm with LARD example | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pruneeuh//POSEIDON/blob/main/tutorials/Tutorial_2.ipynb)            |


# Project Structure :
```
📂 poseidon/                    # Main source director*

├── 📁 numpy/                   # Numpy-related implementations
│   ├── 📁 p3p/
│   │   ├── 📄 p3p.py           # Implements the P3P algorithm and selects the best solution
│   ├── 📁 utils/
│   │   ├── 📄 before_p3p.py    # Generate input data needed by P3P
│   │   ├── 📄 initialize_camera_parameters.py
|
├── 📁 torch/                   # Torch-related implementations
│   ├── 📁 p3p/
│   │   ├── 📄 p3p.py           # Implements the P3P algorithm and selects the best solution
│   ├── 📁 utils/
│   │   ├── 📄 before_p3p.py     # Generate input data needed by P3P
│   │   ├── 📄 initialize_camera_parameters.py
|

📂 tests/                        # Unit tests
├── 📄 test_P3P_batch.py
├── 📄 test_P3P_numpy.py
├── 📄 test_P3P_torch.py

📂 tutorials/                   # Tutorials
├── 📄 Tutorial_1_P3P.ipynb
├── 📄 Tutorial_2_P3P_LARD.ipynb
│   ├── 📁 datas/


```
# Dependencies
POSEIDON relies primarily on:
  - Python ≥ 3.8
  - PyTorch ≥ 1.10 (for tensor operations and autograd)
  - NumPy (for testing and reference comparisons)
  - Pytest (for running the test suite, only for contributors)

All dependencies are declared in the pyproject.toml.

# Tests
A comprehensive test suite is provided in the tests/ folder to ensure the reliability and robustness of the root calculations.
Tests are written using pytest and utilize numpy for result verification (especially for comparison with reference values or handling tolerances).

# 🚀 Installation
**A compléter**


## 👍 Contributing

#To contribute, you can open an
#[issue](https://github.com/Pruneeuh/POSEIDON/issues), or fork this
#repository and then submit changes through a
#[pull-request](https://github.com/Pruneeuh/POSEIDON/pulls).
We use [black](https://pypi.org/project/black/) to format the code and follow PEP-8 convention.
To check that your code will pass the lint-checks, you can run:

```python
tox -e py39-lint
```

You need [`tox`](https://tox.readthedocs.io/en/latest/) in order to
run this. You can install it via `pip`:

```python
pip install tox
```

## 🙏 Acknowledgments

<div align="right">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://share.deel.ai/apps/theming/image/logo?useSvg=1&v=10"  width="25%" align="right">
    <source media="(prefers-color-scheme: light)" srcset="https://www.deel.ai/wp-content/uploads/2021/05/logo-DEEL.png"  width="25%" align="right">
    <img alt="DEEL Logo" src="https://www.deel.ai/wp-content/uploads/2021/05/logo-DEEL.png" width="25%" align="right">
  </picture>
  <picture>
    <img alt="ANITI Logo" src="https://aniti.univ-toulouse.fr/wp-content/uploads/2023/06/Capture-decran-2023-06-26-a-09.59.26-1.png" width="25%" align="right">
  </picture>
</div>
This project received funding from the French program within the <a href="https://aniti.univ-toulouse.fr/">Artificial and Natural Intelligence Toulouse Institute (ANITI)</a>. The authors gratefully acknowledge the support of the <a href="https://www.deel.ai/"> DEEL </a> project.


## 🗞️ Citation



## 📝 License

The package is released under <a href="https://choosealicense.com/licenses/mit"> MIT license</a>.
