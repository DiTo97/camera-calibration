# Camera Calibration using DLT

An example of camera calibration using direct linear transformation
([DLT](https://en.wikipedia.org/wiki/direct_linear_transformation)).

The initial DLT estimate is updated with geometric error minimization using non-linear optimization.

For a written explanation of DLT and the code, see N. Krishna's original [blog post](https://towardsdatascience.com/camera-calibration-with-example-in-python-5147e945cdeb).

![pinhole camera model](images/pinhole-camera-model.png)

## Setup

```sh
git clone https://github.com/DiTo97/camera-calibration.git
cd camera-calibration
```

```sh
python -m venv venv
source venv/bin/activate
```

```sh
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Then, just play with `camera-calibration.ipynb` on Jupyter
