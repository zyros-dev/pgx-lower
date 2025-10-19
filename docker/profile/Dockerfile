# Profile container with PostgreSQL compiled with debug symbols + profiling tools

FROM ubuntu:24.04

ARG KERNEL_VERSION
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
ENV PG_VERSION=17.6
ENV LLVM_VERSION=20

RUN apt-get update && apt-get install -y \
    libssl3 \
    libreadline8 \
    libicu74 \
    libzstd1 \
    zlib1g \
    locales \
    ca-certificates \
    wget \
    gosu \
    sudo \
    software-properties-common \
    llvm-${LLVM_VERSION}-runtime \
    libmlir-${LLVM_VERSION} \
    libelf1 \
    libdw1 \
    libdebuginfod1 \
    libunwind8 \
    libnuma1 \
    libslang2 \
    libperl5.38 \
    libpython3.12 \
    && rm -rf /var/lib/apt/lists/* \
    && locale-gen en_US.UTF-8 \
    && update-locale LANG=en_US.UTF-8

RUN add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
        libpython3.13 \
        libtraceevent1 \
        libjson-c5 \
        libbrotlicommon1 \
        libbrotlidec1 \
        libbz2-1.0 \
        libcom-err2 \
        libffi8 \
        libgmp10 \
        libgnutls30t64 \
        libhogweed6t64 \
        libidn2-0 \
        libnettle8t64 \
        libp11-kit0 \
        librtmp1 \
        libssh2-1t64 \
        libtasn1-6 \
        libunistring5 \
    && rm -rf /var/lib/apt/lists/*

COPY --chown=root:root docker/perf-host /usr/bin/perf

RUN useradd -m -d /var/lib/postgresql -s /bin/bash postgres && \
    mkdir -p /var/lib/postgresql/data && \
    chown -R postgres:postgres /var/lib/postgresql

COPY postgres-debug /usr/local/pgsql
RUN chown -R root:root /usr/local/pgsql

COPY docker/profile/docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh


ENV PATH="/usr/local/pgsql/bin:/usr/lib/llvm-${LLVM_VERSION}/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/pgsql/lib:/usr/lib/llvm-${LLVM_VERSION}/lib:${LD_LIBRARY_PATH}"

WORKDIR /workspace
EXPOSE 5432

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
