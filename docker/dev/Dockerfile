FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LLVM_VERSION=20
ENV PG_VERSION=17.5

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    wget \
    curl \
    bc \
    bison \
    flex \
    python3 \
    python3-pip \
    python3-venv \
    lsb-release \
    software-properties-common \
    gnupg \
    zlib1g-dev \
    libreadline-dev \
    libssl-dev \
    libicu-dev \
    pkg-config \
    libgtest-dev \
    libzstd-dev \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    ./llvm.sh ${LLVM_VERSION} all && \
    rm llvm.sh && \
    update-alternatives --install /usr/bin/clang clang /usr/lib/llvm-${LLVM_VERSION}/bin/clang 100 && \
    update-alternatives --install /usr/bin/clang++ clang++ /usr/lib/llvm-${LLVM_VERSION}/bin/clang++ 100 && \
    update-alternatives --install /usr/bin/cc cc /usr/lib/llvm-${LLVM_VERSION}/bin/clang 100 && \
    update-alternatives --install /usr/bin/c++ c++ /usr/lib/llvm-${LLVM_VERSION}/bin/clang++ 100

RUN apt-get update && apt-get install -y \
        libmlir-${LLVM_VERSION}-dev \
        mlir-${LLVM_VERSION}-tools \
        llvm-${LLVM_VERSION}-dev \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://ftp.postgresql.org/pub/source/v${PG_VERSION}/postgresql-${PG_VERSION}.tar.gz && \
    tar -xzf postgresql-${PG_VERSION}.tar.gz && \
    cd postgresql-${PG_VERSION} && \
    ./configure --prefix=/usr/local/pgsql --with-openssl --with-icu && \
    make -j$(nproc) && \
    make install && \
    cd / && \
    rm -rf postgresql-${PG_VERSION} postgresql-${PG_VERSION}.tar.gz

RUN useradd -m -d /var/lib/postgresql -s /bin/bash postgres && \
    mkdir -p /var/lib/postgresql/data && \
    chown -R postgres:postgres /var/lib/postgresql

USER postgres
RUN /usr/local/pgsql/bin/initdb -D /var/lib/postgresql/data

USER root

ENV PATH="/usr/local/pgsql/bin:/usr/lib/llvm-${LLVM_VERSION}/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/pgsql/lib:/usr/lib/llvm-${LLVM_VERSION}/lib:${LD_LIBRARY_PATH}"
ENV MLIR_DIR="/usr/lib/llvm-${LLVM_VERSION}/lib/cmake/mlir"
ENV LLVM_DIR="/usr/lib/llvm-${LLVM_VERSION}/lib/cmake/llvm"

WORKDIR /workspace

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 5432

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
