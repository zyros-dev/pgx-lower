# Slim benchmark container using pre-built release extension
#
# Expects: build-docker-release/extension/pgx_lower.so in project root
#          (Run: cd docker && make build-benchmark)
#
# Build context: Project root (../)

# Stage 1: PostgreSQL binaries from dev image
FROM pgx-lower:dev AS pgbuilder

# Stage 2: Runtime - Slim image with PostgreSQL + extension
FROM ubuntu:24.04

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
    llvm-${LLVM_VERSION}-runtime \
    libmlir-${LLVM_VERSION} \
    && rm -rf /var/lib/apt/lists/* \
    && locale-gen en_US.UTF-8 \
    && update-locale LANG=en_US.UTF-8

RUN wget -q https://github.com/janestreet/magic-trace/releases/latest/download/magic-trace -O /usr/local/bin/magic-trace && \
    chmod +x /usr/local/bin/magic-trace

RUN wget -q https://get.perfetto.dev/traceconv -O /usr/local/bin/traceconv && \
    chmod +x /usr/local/bin/traceconv

RUN useradd -m -d /var/lib/postgresql -s /bin/bash postgres && \
    mkdir -p /var/lib/postgresql/data && \
    chown -R postgres:postgres /var/lib/postgresql

COPY --from=pgbuilder /usr/local/pgsql /usr/local/pgsql

COPY docker/benchmark/docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENV PATH="/usr/local/pgsql/bin:/usr/lib/llvm-${LLVM_VERSION}/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/pgsql/lib:/usr/lib/llvm-${LLVM_VERSION}/lib:${LD_LIBRARY_PATH}"

WORKDIR /workspace
EXPOSE 5432

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
