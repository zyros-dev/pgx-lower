FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LLVM_VERSION=20
ENV PG_VERSION=17.6
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

RUN apt-get update && apt-get install -y \
    build-essential \
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
    locales \
    sudo \
    && rm -rf /var/lib/apt/lists/* \
    && locale-gen en_US.UTF-8 \
    && update-locale LANG=en_US.UTF-8

RUN wget -q https://github.com/janestreet/magic-trace/releases/latest/download/magic-trace -O /usr/local/bin/magic-trace && \
    chmod +x /usr/local/bin/magic-trace

RUN wget https://github.com/Kitware/CMake/releases/download/v3.31.6/cmake-3.31.6-linux-x86_64.sh && \
    chmod +x cmake-3.31.6-linux-x86_64.sh && \
    ./cmake-3.31.6-linux-x86_64.sh --skip-license --prefix=/usr/local && \
    rm cmake-3.31.6-linux-x86_64.sh

RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test && \
    apt-get update && \
    apt-get install -y gcc-14 g++-14 libstdc++-14-dev && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-14 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-14 100 && \
    rm -rf /var/lib/apt/lists/*

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
        libc++-${LLVM_VERSION}-dev \
        libc++abi-${LLVM_VERSION}-dev \
    && rm -rf /var/lib/apt/lists/* \
    && mv /usr/lib/llvm-${LLVM_VERSION}/include/cxxabi.h /usr/lib/llvm-${LLVM_VERSION}/include/cxxabi.h.bak

RUN wget https://ftp.postgresql.org/pub/source/v${PG_VERSION}/postgresql-${PG_VERSION}.tar.gz && \
    tar -xzf postgresql-${PG_VERSION}.tar.gz && \
    cd postgresql-${PG_VERSION} && \
    CC=clang CXX=clang++ ./configure --prefix=/usr/local/pgsql --with-openssl --with-icu && \
    make -j$(nproc) && \
    make install && \
    cd / && \
    rm -rf postgresql-${PG_VERSION} postgresql-${PG_VERSION}.tar.gz


RUN useradd -m -d /var/lib/postgresql -s /bin/bash postgres && \
    mkdir -p /var/lib/postgresql/data && \
    chown -R postgres:postgres /var/lib/postgresql

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
