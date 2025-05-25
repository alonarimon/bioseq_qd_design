#!/usr/bin/env bash

# Parse arguments
OPTS=$(getopt -o w -l writable -- "$@")
if [ $? != 0 ]; then echo "Failed to parse arguments." >&2 ; exit 1 ; fi
eval set -- "$OPTS"

# Make sure WANDB_API_KEY is set before launching
if [[ -z "${WANDB_API_KEY}" ]]; then
	echo "Warning: WANDB_API_KEY is not set in your environment." >&2
fi

# Extract options
writable_flag=""
while true ; do
	case "$1" in
		-w | --writable )
			writable_flag="--writable"; shift ;;
		-- )
			shift; break ;;
		* )
			break ;;
	esac
done

# Shell into the container
apptainer shell \
	--bind $(pwd):/workdir/ \
	--env WANDB_API_KEY="${WANDB_API_KEY}"\
	--env CUDA_VISIBLE_DEVICES="2" \
	--cleanenv \
	--containall \
	--home /tmp/ \
	--no-home \
	--nv \
	--pwd /workdir/ \
	--workdir apptainer/ \
	$writable_flag \
	apptainer/container.sif

