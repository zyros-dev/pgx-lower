# Profile container with PostgreSQL compiled with debug symbols + profiling tools

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
ENV PG_VERSION=17.6
ENV LLVM_VERSION=20

# Install runtime dependencies + profiling tools
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
    # LLVM runtime libraries for JIT
    llvm-${LLVM_VERSION}-runtime \
    libmlir-${LLVM_VERSION} \
    # Profiling tools
    linux-tools-generic \
    linux-tools-common \
    && rm -rf /var/lib/apt/lists/* \
    && locale-gen en_US.UTF-8 \
    && update-locale LANG=en_US.UTF-8

# Create postgres user
RUN useradd -m -d /var/lib/postgresql -s /bin/bash postgres && \
    mkdir -p /var/lib/postgresql/data && \
    chown -R postgres:postgres /var/lib/postgresql

# PostgreSQL and extension will be copied at runtime from /workspace
# This allows updating without rebuilding the Docker image

ENV PATH="/usr/local/pgsql/bin:/usr/lib/llvm-${LLVM_VERSION}/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/pgsql/lib:/usr/lib/llvm-${LLVM_VERSION}/lib:${LD_LIBRARY_PATH}"

WORKDIR /workspace
EXPOSE 5432

# Keep container alive for manual PostgreSQL startup
CMD ["tail", "-f", "/dev/null"]
