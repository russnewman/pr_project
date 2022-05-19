# Final project for the Perception in Robotics course

This repository contains the initial stage of the article [gradSLAM: Automagically differentiable SLAM](https://arxiv.org/pdf/1910.10672v3.pdf) implementation.

To run the project you need to install all necessary modules:

```bash
pip install -r requirements.txt
```

To test differentiable Levenberg-Marquardt (LM) solver you can run:

```bash
python3 gradLM_exp.py
python3 gradLM_sin.py
```

or create your own fuction with `Function` interface (see `GradLM.py` module).

Also we provide `PoseEstimation` class for further steps.
