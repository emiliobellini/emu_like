# emu_like

<!-- ## Description -->

`emu_like` is a library created to build and use emulators for cosmological likelihoods.

As described below, it allows to generate a sample of data, and use it to train an emulator with a user-specified architecture. In [Sampling functions](src/emu_like/sampling_functions.py) it is possible to find all the functions that can be sampled. On top of simple functions that can be used to familiarise with the code, it has an interface with the [Cobaya](https://cobaya.readthedocs.io/en/latest/) sampler, which allows to easily build an emulator for any of the likelihoods implemented in it.

The code can be used either as a Python module or as a standalone code. As a standalone code there are three main pipelines:
- [sample](pipelines/sample.py): to generate and store a sample of data that will be used to train the emulator;
- [train](pipelines/train.py): to train the emulator on sampled data and store it;
- [mcmc](pipelines/mcmc.py): this is used to run an mcmc with an emulator.

As a Python module the two main classes that should be called are:
- [Sample](src/emu_like/sample.py): to generate, load and save samples of data;
- [Emulator](src/emu_like/emu.py): to train, load and save emulators. While the Emulator class is the base class of emulators, it is often convenient to call directly the type of emulator needed, e.g. [FFNNEmu](src/emu_like/ffnn_emu.py).

Each emulator returns likelihood values (depending on the number of likelihoods we emulate) providing a list of cosmological parameters.

**NOTE**: this code has a modular structure and can be generalised in a relatively simple way. Adding new samplers, scalers, emulators, loss functions (or anthing else) to the code can be done by simply copying a similar object and making the changes needed.


## Installation

To install the code, download it or clone it from github with
```
git clone https://github.com/emiliobellini/emu_like.git
```
Then, install it from source with
```
cd emu_like/
python -m pip install .
```
This will take care of installing all the dependencies.

PS: we recommend to use it with a virtual environment, running
```
python -m venv /path/to/new/virtual/environment
source /path/to/new/virtual/environment/bin/activate
```


## Usage

If you plan to create your own pipeline, follow examples in [this folder](examples). They explain how to get a sample, train and use an emulator in simple scenarios.

The sample pipeline can be run from the main folder with
```
python emu_like.py sample params.yaml -v
```
Ready to use parameter files can also be found in the main directory
- `simple_sample.yaml`: to sample a straight line in 1D on a grid;
- `planck_sample.yaml`: to sample the Planck likelihood using Cobaya, classy and standard cosmological parameters.

The train pipeline can be run from the main folder with
```
python emu_like.py train params.yaml -v -p
```
Ready to use parameter files can also be found in the main directory
- `simple_train.yaml`: to train an emulator that fits a straight line in 1D;
- `planck_train.yaml`: to train an emulator for the Planck total likelihood.

Finally, it is possible to test the emulator created with
```
python emu_like.py mcmc planck_mcmc.yaml -v
```
where `planck_mcmc.yaml` contains all the information needed by the mcmc sampler.

**NOTE**: all the parameter files mentioned here can be used as a guidance to understand the input parameters at each step and their usage.


## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Contributing
For bugs and feature requests consider using the [issue tracker](https://github.com/emiliobellini/emu_like/issues).

## License
`emu_like` is released under the GPL-3 license (see [LICENSE](LICENSE)).
