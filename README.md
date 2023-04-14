# Camera Calibration using DLT

A camera calibration example using direct linear transformation
([DLT](https://en.wikipedia.org/wiki/direct_linear_transformation))
and geometric error minimization with non-linear optimization.

For a written explanation of the code, see N. Krishna's original [blog post](https://towardsdatascience.com/camera-calibration-with-example-in-python-5147e945cdeb).

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
python -m pip install -r requirements.txt
```
