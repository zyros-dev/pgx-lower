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
    # LLVM runtime libraries for JIT
    llvm-${LLVM_VERSION}-runtime \
    libmlir-${LLVM_VERSION} \
    && rm -rf /var/lib/apt/lists/* \
    && locale-gen en_US.UTF-8 \
    && update-locale LANG=en_US.UTF-8

# Create postgres user
RUN useradd -m -d /var/lib/postgresql -s /bin/bash postgres && \
    mkdir -p /var/lib/postgresql/data && \
    chown -R postgres:postgres /var/lib/postgresql

COPY --from=pgbuilder /usr/local/pgsql /usr/local/pgsql

# Copy pgx-lower extension from build context (host filesystem after build-release.sh)
# NOTE: Docker caches COPY layers based on file timestamps/content
# When pgx_lower.so changes, this layer and all subsequent layers will rebuild
COPY build-docker-release/extension/pgx_lower.so /usr/local/pgsql/lib/pgx_lower.so
COPY extension/pgx_lower.control /usr/local/pgsql/share/extension/pgx_lower.control
COPY extension/sql/pgx_lower--1.0.sql /usr/local/pgsql/share/extension/pgx_lower--1.0.sql

# Set up environment
ENV PATH="/usr/local/pgsql/bin:/usr/lib/llvm-${LLVM_VERSION}/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/pgsql/lib:/usr/lib/llvm-${LLVM_VERSION}/lib:${LD_LIBRARY_PATH}"

# Initialize database as postgres user
USER postgres
RUN /usr/local/pgsql/bin/initdb -D /var/lib/postgresql/data

USER root

# Copy configuration files
COPY docker/benchmark/postgresql.conf /var/lib/postgresql/postgresql.conf
COPY docker/benchmark/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && \
    chown postgres:postgres /var/lib/postgresql/postgresql.conf

WORKDIR /workspace
EXPOSE 5432

ENTRYPOINT ["/entrypoint.sh"]
CMD ["postgres"]
