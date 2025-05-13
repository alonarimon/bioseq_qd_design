# HPC utils

`hpc.sh` is a script designed to simplify:
- the generation of jobscripts.
- running jobscripts on the HPC.

## Prerequisites

- Go at the root of the repository.
- A container created with [`build_container.sh`](https://gitlab.doc.ic.ac.uk/AIRL/airl_tools/container-utils) exists.
- There is a file named `apptainer/hpc.yaml` containing the parameters of the jobscripts you want to generate and run on the HPC.

## How to use `hpc.sh`?

### Usage

```bash
hpc.sh <container_path>
```

For example,
```bash
hpc.sh apptainer/container_2023-10-04_174826_a04b5a55a7e22f715d4b0eb1f35447cd20f86dd3.sif
```

The script will:
1. set up a directory for the experiment on the HPC, called `~/projects/<repository_name>/output/<container_name>/`.
2. send the container to the HPC.
3. send a file containing the output of the git log command to the HPC.
4. generate jobscripts according to the configuration supplied in `apptainer/hpc.yaml` and using `template.job`.
5. send and submit the jobs to the HPC.

### Set up SSH config

To avoid entering your credentials multiple times during the execution of the script `hpc.sh`, you can set up your SSH config by adding the following:
```
Host hpc cx3 login.hpc.ic.ac.uk login.hpc.imperial.ac.uk
        HostName login.hpc.imperial.ac.uk
        ControlPath ~/.ssh/controlmasters/%r@%h:%p
        ControlMaster auto
        ControlPersist yes
        User <username>

Host hx1 login.hx1.hpc.ic.ac.uk login.hx1.hpc.imperial.ac.uk
        HostName login.hx1.hpc.imperial.ac.uk
        ControlPath ~/.ssh/controlmasters/%r@%h:%p
        ControlMaster auto
        ControlPersist yes
        User <username>
```

## HPC

For more information about the HPC, you can check the [website](https://www.imperial.ac.uk/computational-methods/hpc/) or the [wiki](https://wiki.imperial.ac.uk/display/HPC/High+Performance+Computing).
