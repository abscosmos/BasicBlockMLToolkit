FROM ubuntu:22.04

# avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# install common development tools and programs for tracing
RUN apt-get update && apt-get install -y \
    # core system utilities
    coreutils findutils grep sed gawk \
    # file operations
    tar gzip zip unzip \
    # network tools
    curl wget \
    # development tools
    build-essential git \
    gcc g++ make cmake pkg-config \
    # interpreters/runtimes
    python3 nodejs npm \
    # text editors
    vim nano \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
RUN mkdir -p traces work