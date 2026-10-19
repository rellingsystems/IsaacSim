![Isaac Sim](docs/readme/hero_shot_compressed.png)

---
# Isaac Sim

## Relling development platform

The Relling fork lives at <https://github.com/rellingsystems/IsaacSim>. Develop on
`relling/dev`; keep `upstream` pointed at <https://github.com/isaac-sim/IsaacSim>.
The initial baseline is upstream `main` at
`7c206f75bdadd9e05fc457f19863ca4c3f0cb693` (6.1.0 GA; the upstream `VERSION`
file reads `6.1.0-rc.26`). Upstream's default `develop` branch is newer; update
the baseline deliberately and repeat the smoke test after an upgrade.

Use the Mac for editing and Stallion for native Linux x86_64 source builds and
GPU execution. A public binary installation does not validate changes to this
fork. Do not treat the platform as GPU-verified until a Stallion run produces
a passing `acceptance.json` and its associated `physics.json`.

Validated on Stallion on 2026-10-19: Ubuntu 24.04.4, RTX 4090 (24 GiB), driver
580.178.04, and GCC 11. The release source build passed in 1661 seconds. The
headless smoke at commit `7713f3ca3db33e31a941633a2f81a95a0a250898` passed
240 physics steps and eight render calls; the cube settled at 0.25 m. The warm
run reported app ready at 7.04 seconds and exited successfully. Evidence is in
Stallion's `_relling_runs/1792429042852699778-build/` and
`_relling_runs/1792431126764803077-smoke/`. This is a smoke result, not a
performance benchmark or validation of GUI, ROS, robot, or sensor workflows.

```bash
ssh stallion
mkdir -p ~/src/relling
cd ~/src/relling
git clone --branch relling/dev https://github.com/rellingsystems/IsaacSim.git
cd IsaacSim
git remote add upstream https://github.com/isaac-sim/IsaacSim.git
# GCC 11 is installed alongside the host compiler. Select it for this shell.
mkdir -p _relling_toolchain/bin
ln -sf /usr/bin/gcc-11 _relling_toolchain/bin/gcc
ln -sf /usr/bin/g++-11 _relling_toolchain/bin/g++
export PATH="$PWD/_relling_toolchain/bin:$PATH"
export CC=gcc CXX=g++
python3 tools/relling/dev.py preflight
```

Preflight inventories the selected GPU, driver, compiler, free RAM, and disk.
It requires GCC/G++ 11, Git LFS, an idle GPU with at least 16 GiB VRAM, 16 GiB
available RAM for building (8 GiB for the lightweight smoke), and a conservative
100 GiB free build reserve. The default is
two build jobs so Stallion's 32 GiB RAM can also support its existing services.
These checks are readiness gates; they do not replace NVIDIA's RTX and driver
compatibility requirements below. Install missing host prerequisites using the
upstream instructions. Use a private compiler PATH shim if changing the
machine-wide compiler would affect other projects. No host packages, drivers,
Docker settings, or system compiler alternatives are changed by our helper.

```bash
# Review NVIDIA's terms interactively; the helper never accepts them for you.
./tools/eula_check.sh
python3 tools/relling/dev.py build --jobs 2
python3 tools/relling/dev.py smoke --gpu 0

# Interactive GUI when a desktop session is available:
./_build/linux-x86_64/release/isaac-sim.sh
```

Build uses the upstream source-build path (`--release --jobs 2`) and pulls
Git LFS assets. This baseline disables Docker in `repo.toml`; its build parser
does not accept `--no-docker` despite older build notes mentioning that flag.
Smoke launches the built checkout headlessly, requests
single-GPU PhysX, renders eight frames, and requires a procedural 0.5 m cube
to fall from 2 m and settle at 0.25 m on a floor over 240 physics steps. It does
not require remote robot assets. This proves basic initialization and physics;
robot controllers, cameras, ROS, streaming, and performance require separate
tests. A GPU request alone does not prove that the engine avoided CPU fallback;
inspect `smoke.log` alongside the result before making that claim.

Run custom Python scripts with `bash tools/relling/python.sh /absolute/path/to/script.py`.
This wrapper puts the build's bundled NVIDIA runtime libraries ahead of older
system CUDA libraries. Smoke uses it with `--no-ros-env` and launches from the
release directory. This avoids a CUDA 12.0 system runtime shadowing the CUDA
12.8 runtime needed by the pinned PyTorch build; the host CUDA installation is
left intact.

Kit also needs a user-writable `~/Documents/Kit` directory for its normal app
data. This directory was created for Stallion's user because its parent
`Documents` directory is owned by root. Smoke uses Kit's default fast shutdown;
the controller requires a zero process exit, valid physics results, and no
fatal error or traceback before it writes a passing acceptance record.

Each build or smoke attempt gets a fresh `_relling_runs/<id>-<action>/` with
host inventory, source commit and dirty status, logs, an active PID file, and
`acceptance.json`. Existing runs are never overwritten. The build deadline is
three hours; smoke has a ten-minute deadline including first-start shader work.
The helper kills the process group on timeout. Watch logs from a second SSH
session while it runs; it writes acceptance only after the process exits.
Run IDs and runtime data stay out of Git. GPU jobs already using more than
2 GiB VRAM or 20% utilization block the helper; coordinate with their owner.

For development, edit under `source/`, repeat the release build, and rerun smoke.
Keep custom simulation scripts under `tools/relling/` initially. Use Onshape and
editable FeatureScript for custom CAD; preserve supplied vendor CAD and import
exports into simulation without changing the CAD source of truth.

To update the fork, fetch `upstream`, merge the desired upstream release into
`relling/dev`, resolve conflicts, then repeat preflight, build, and smoke before
pushing. Keep upstream workflows disabled unless runner labels and secrets have
been reviewed for the fork; this bootstrap does not register Stallion as a
GitHub Actions runner.

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://docs.python.org/3/whatsnew/3.12.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![Linux aarch64 platform](https://img.shields.io/badge/platform-linux--aarch64-orange.svg)](https://docs.nvidia.com/dgx/dgx-os-7-user-guide/introduction.html)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](LICENSE)

NVIDIA Isaac Sim™ is a simulation platform built on NVIDIA Omniverse, designed to develop, test, train, and deploy AI-powered robots in realistic virtual environments. It supports importing robotic systems from common formats such as URDF, MJCF, and CAD. The simulator leverages high-fidelity, GPU-accelerated physics engines to simulate accurate dynamics and support multi-sensor RTX rendering at scale. It comes equipped with end-to-end workflows including synthetic data generation, reinforcement learning, ROS integration, and digital twin simulation. Isaac Sim provides the infrastructure needed to support robotics development at any stage.

## Key Features

- [Asset Import & Export](https://docs.isaacsim.omniverse.nvidia.com/latest/importer_exporter/importers_exporters.html): Importing and exporting robots and environments from and to non-USD format.
- [Robot Tuning](https://docs.isaacsim.omniverse.nvidia.com/latest/robot_setup/index.html): Optimize robot for physics accuracy, computation efficiency, or photorealism
- [Robot Simulation](https://docs.isaacsim.omniverse.nvidia.com/latest/robot_simulation/index.html): Tools for moving robots, such as controllers, motion generation and kinematics solvers, and policy integration.
- [Sensors](https://docs.isaacsim.omniverse.nvidia.com/latest/sensors/index.html): RTX and physics-based sensors

## Key Applications

- [Isaac Lab](https://docs.isaacsim.omniverse.nvidia.com/latest/isaac_lab_tutorials/index.html): GPU-accelerated framework built for reinforcement learning, imitation learning, and motion planning.
- [ROS Bridge](https://docs.isaacsim.omniverse.nvidia.com/latest/ros2_tutorials/ros2_landing_page.html): Integration with Robot Operating System (ROS).
- [Synthetic Data Generation](https://docs.isaacsim.omniverse.nvidia.com/latest/synthetic_data_generation/index.html): Collection of SDG tools

## Documentation

For the latest Isaac Sim documentation, see [Isaac Sim Documentation](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html).
Follow these links to get started:

- [Tutorials](https://docs.isaacsim.omniverse.nvidia.com/latest/introduction/quickstart_index.html)
- [Assets](https://docs.isaacsim.omniverse.nvidia.com/latest/assets/usd_assets_overview.html)


## Prerequisites and Environment Setup

Ensure your system is set up with the following before building Isaac Sim:

- **Operating System**: Windows 11 or Linux (Ubuntu 22.04/24.04)

  > **(Linux) Ubuntu 24.04**
  > Building with Ubuntu 24.04 requires GCC/G++ 11 to be installed, GCC/G++ 12+ is not supported.

- **GPU**: For additional information on GPU features and requirements, see [NVIDIA GPU Requirements](https://docs.omniverse.nvidia.com/dev-guide/latest/common/technical-requirements.html)

  #### Local Workstation

  | Min | Recommended | Best |
  |-----|-------------|------|
  | RTX 4080 | RTX 5080 | RTX PRO 6000 Blackwell Workstation |
  |  | RTX 5880 Ada | RTX PRO 5000 Blackwell Workstation |

  #### Datacenter

  | Min | Recommended | Best |
  |-----|-------------|------|
  | A40 | L40S | RTX PRO 6000 Blackwell Server |
  |  | L20 | |

- **Driver**: See [NVIDIA Driver Requirements](https://docs.omniverse.nvidia.com/dev-guide/latest/common/technical-requirements.html)

- **Internet Access**: Required for downloading the Omniverse Kit SDK, extensions, and tools. Allow outbound HTTPS
  access to `ovextensionsprod.blob.core.windows.net` so the build can resolve public Kit and Isaac Sim extensions.



### Required Software Dependencies

- [**Git**](https://git-scm.com/downloads): For version control and repository management

- [**Git LFS**](https://git-lfs.com/): For managing large files within the repository

- **(Windows - C++ Only) Microsoft Visual Studio 2022**:

- Install Visual Studio 2022, Windows SDK, and MSVC using Winget by running the following command in PowerShell:

  ```powershell
  winget install --id=Microsoft.VisualStudio.2022.Community -e --override "--add Microsoft.VisualStudio.Workload.NativeDesktop --includeRecommended"
  ```
  
  [Additional information on Windows development configuration](docs/readme/windows_developer_configuration.md)


- **(Linux) build-essentials**: A package that includes `make` and other essential tools for building applications.  For Ubuntu, install with:

  ```bash
  sudo apt-get install build-essential
  ```

  > **(Linux) ⚠️**
  > Please use GCC/G++ 11, higher versions are not supported yet. To install GCC/G++ 11, run the following commands:
  > ```bash
  > sudo apt-get install gcc-11 g++-11
  > sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 200
  > sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 200
  > ```

  > **(Linux aarch64) ⚠️**
  > On aarch64 hosts (e.g. DGX Spark), X11 development headers are required to build Python packages that lack pre-built wheels:
  > ```bash
  > sudo apt-get install -y libx11-dev xorg-dev
  > ```

  > **Compiler Version Check ⚠️**
  > We have added a version checker to our build process. If you do not have the default versions you are still able to execute a build, add  `--skip-compiler-version-check` to `build.[sh/bat]` when building.  Proceed at your own risk, unsupported build environments may encounter build and runtime issues.

### Recommended Software

- [**(Linux) Docker**](https://docs.docker.com/engine/install/ubuntu/): For containerized development and deployment. **Ensure non-root users have Docker permissions.**

- [**(Linux) NVIDIA Container Toolkit**](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html): For GPU-accelerated containerized development and deployment. **Installation and Configuring Docker steps are required.**

- [**VSCode**](https://code.visualstudio.com/download) (or your preferred IDE): For code editing and development

## Quick Start

This section guides you through building Isaac Sim from source code.

### 1. Clone the Repository


```bash
git clone -b main https://github.com/isaac-sim/IsaacSim.git isaacsim
cd isaacsim
git lfs install
git lfs pull
```

### 2. Build

Run the following command to initiate the configuration wizard:

**Linux:**

Confirm that GCC/G++ 11 is being used before building using the following commands:

```bash
gcc --version
g++ --version
```

```bash
./build.sh
```

**Windows:**

> **⚠️ Windows Path Length Limitation**
> Windows has a path length limitation of 260 characters. If you encounter errors related missing files or other build errors, try moving the repository to a shorter path.

```powershell
build.bat
```

### 3. Run

> **⚠️ Startup Time**
> The first time loading Isaac Sim may take up to several minutes as Extensions and Shader are loaded and cached. The subsequent startup time should be in the ranges of 10-30 seconds depending on hardware configuration.



Navigate to the corresponding binary directory for your platform and run the executable.

**Linux (x86_64):**
```bash
cd _build/linux-x86_64/release
./isaac-sim.sh
```

**Linux (aarch64):**
```bash
cd _build/linux-aarch64/release
./isaac-sim.sh
```

**Windows:**
```powershell
cd _build/windows-x86_64/release
isaac-sim.bat
```

> NOTE: If this is your first time building Isaac Sim, you will be prompted to accept the Omniverse Licensing Terms.



## Advanced Build Options


Isaac Sim uses a custom build system with the following key options:


### Core Build Options
- `-c, --clean`: Clean the repository and exit
- `-x, --rebuild`: Clean the repository before building (full rebuild)
- `-h, --help`: Show all available build options


### Configuration Options
- `--config [debug|release]`: Specify build configuration (default: both)
- `-d, --debug`: Build only debug configuration
- `-r, --release`: Build only release configuration


### Advanced Options
- `-j NUM_CORES, --jobs NUM_CORES`: Limit the number of parallel compilation jobs
- `-v, --verbose`: Enable verbose build output
- `-q, --quiet`: Suppress build output


### Build Steps Control
- `--fetch-only`: Only fetch dependencies and stop
- `-g, --generate`: Generate projects, stage files and stop
- `-s, --stage`: Stage files, skip generation step
- `-b, --build-only`: Only perform building step, skip others
- `--post-build-only`: Only perform post-build step

## Usage
Congratulations on installing Isaac Sim! To get started with using Isaac Sim, follow these [Quick Tutorials](https://docs.isaacsim.omniverse.nvidia.com/latest/introduction/quickstart_index.html). For more information, visit our full [documentation](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html).

## Additional Build Tools

Beyond building and running from source (see [Quick Start](#quick-start)), Isaac Sim can also be packaged as a standalone binary archive, built as Python wheels, or deployed as a Docker container.

### Binary Package

Build a standalone redistributable binary package from source. A successful [build](#quick-start) is required before packaging.

**Linux:**

```bash
./repo.sh package --config release -m isaac-sim-standalone
```

> **Note:** The same command works on both x86_64 and aarch64 hosts. The build system detects the platform automatically.

**Windows:**

```powershell
.\repo.bat package --config release -m isaac-sim-standalone
```

The packaged archive is written to the `_build/packages/` directory.

### PIP Packages

Build Isaac Sim Python (PIP) wheels locally from a successful [build](#quick-start). Pre-built wheels for released versions are also available on [pypi.nvidia.com](https://pypi.nvidia.com); see [Install Isaac Sim using PIP](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_python.html#installation-using-pip) for the install-only path.

A successful [build](#quick-start) is required before packaging.

**Linux:**

```bash
./repo.sh python_package --create
./repo.sh comment_archive_deps
./repo.sh python_package --wheel
```

**Windows:**

```powershell
.\repo.bat python_package --create
.\repo.bat comment_archive_deps
.\repo.bat python_package --wheel
```

The three steps, in order:

1. `python_package --create` stages per-package source trees under `_build/packages/python/` from the wheel definitions in [python_packages.toml](python_packages.toml).
2. `comment_archive_deps` comments out references to Kit pip-archive extensions (`omni.kit.pip_archive`, `omni.isaac.core_archive`, `omni.isaac.ml_archive`, `omni.pip.compute`, `omni.pip.cloud`, `isaacsim.pip.newton`, `isaacsim.pip.nv`, `isaacsim.pip.onnx`) in the generated `extension.toml` and `*.kit` files so wheel metadata is self-contained.
3. `python_package --wheel` builds the `.whl` files into `_build/packages/dist/`.

Install locally-built wheels into a Python 3.12 virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install _build/packages/dist/*.whl
```

> **(Linux aarch64) ⚠️**
> On aarch64 hosts, X11 development headers are required to build transitive Python dependencies that lack pre-built aarch64 wheels (same note as under [Prerequisites and Environment Setup](#prerequisites-and-environment-setup)):
> ```bash
> sudo apt-get install -y libx11-dev xorg-dev
> ```

### Container (Docker)

For building a Docker image, running with Docker Compose, and web-based streaming, see [tools/docker/README.md](tools/docker/README.md).


## Troubleshooting

- Please see the [FAQ](https://docs.isaacsim.omniverse.nvidia.com/latest/overview/faq_index.html), [Troubleshooting](https://docs.isaacsim.omniverse.nvidia.com/latest/overview/troubleshooting.html), and [Known Issues](https://docs.isaacsim.omniverse.nvidia.com/latest/overview/known_issues.html) for common questions, fixes, and workarounds.

- On Linux, if you encounter network connectivity issues when building (such as corporate firewalls), run the following commands:

  ```bash
  export http_proxy="http://{Your IP address}:7890"
  export https_proxy="http://{Your IP address}:7890"
  ```

  - Use these variables only when you use proxy software or are behind a corporate firewall. Replace port 7890 with
    the port used by your proxy.

- If dependency resolution reports `found 0 packages` for a public registry such as `kit/prod/sdk` and exits with
  code 55, verify that the build process can reach `ovextensionsprod.blob.core.windows.net` through your firewall or
  proxy. Reaching the host with a browser or a separate command-line client does not confirm that the build inherited
  the same proxy settings.


## Support

* Please use GitHub [Discussions](https://github.com/isaac-sim/IsaacSim/discussions) for discussing ideas, asking questions, and requests for new features.
* Github [Issues](https://github.com/isaac-sim/IsaacSim/issues) should only be used to track executable pieces of work with a definite scope and a clear deliverable. These can be fixing bugs, documentation issues, new features, or general updates.

## Connect with the NVIDIA Omniverse Community

Have a project or resource you'd like to share more widely? We'd love to hear from you! Reach out to the
NVIDIA Omniverse Community team at OmniverseCommunity@nvidia.com to discuss potential opportunities
for broader dissemination of your work.

## License

Licensing terms can be found in the [License File](LICENSE).

## Citation

To cite Isaac Sim, click on "Cite this repository" in the right sidebar of the [Isaac Sim GitHub repository](https://github.com/isaac-sim/IsaacSim) landing page and select one of the listed citation entries.

## Contributing

We do not support direct community contributions at the moment.
