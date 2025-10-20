.PHONY: up down build-ptest logs shell restart status clean all rebuild
.PHONY: build-debug ptest-debug build-release ptest-release build-profile ptest-profile release-container
.PHONY: build-benchmark up-benchmark down-benchmark shell-benchmark logs-benchmark
.PHONY: build-postgres-debug profile-container clean-dockers bench-dockers bench-list

# Development container targets
up:
	docker compose up -d dev

down:
	docker compose down

restart:
	docker compose restart dev

status:
	docker compose ps

build-ptest:
	@./dev/build-ptest.sh

logs:
	docker logs -f pgx-lower-dev

shell:
	docker exec -it pgx-lower-dev bash

clean:
	docker compose down -v

clean-dockers:
	@echo "Completely removing all Docker containers, volumes, and build artifacts..."
	@echo "Step 1: Stopping and removing all containers and volumes..."
	docker compose down -v --remove-orphans
	@echo "Step 2: Removing Docker images..."
	docker rmi pgx-lower:dev pgx-lower:benchmark pgx-lower:profile 2>/dev/null || true
	@echo "Step 3: Removing build directories..."
	rm -rf ../build-docker-ptest ../build-docker-ptest-release ../postgres-debug
	@echo "Step 4: Listing what remains..."
	@if docker images | grep pgx-lower; then \
		echo "Warning: Some pgx-lower images still exist"; \
	else \
		echo "All pgx-lower images removed"; \
	fi
	@echo "Docker cleanup completed!"

all: down
	@echo "Building all containers and setting up complete environment..."
	@echo "Step 1: Building dev image first (required by benchmark)..."
	docker compose build dev
	@echo "Step 2: Starting dev container..."
	docker compose up -d dev
	@sleep 3
	@echo "Step 3: Building PostgreSQL with debug symbols in dev container..."
	@$(MAKE) build-postgres-debug
	@echo "Step 4: Building debug extension in dev container..."
	@$(MAKE) build-debug
	@echo "Step 5: Building release extension in dev container..."
	@$(MAKE) build-release
	@echo "Step 6: Running regression tests on release build..."
	@$(MAKE) ptest-release
	@echo "Step 7: Building benchmark image..."
	docker compose build benchmark
	@echo "Step 8: Starting benchmark container..."
	docker compose up -d benchmark
	@echo "Step 9: Deploying release extension to benchmark container..."
	@$(MAKE) release-container
	@echo "Step 10: Installing native PostgreSQL for profiling..."
	@$(MAKE) install-host-profile
	@echo "Step 11: Cleaning and starting host PostgreSQL..."
	@pkill -9 -f "^$(PGX_PROFILE_DIR)/bin/postgres" 2>/dev/null || true
	@rm -rf $(PGX_PROFILE_DIR)/data
	@echo "Initializing database in container (avoids host locale issues)..."
	@docker exec -u postgres pgx-lower-dev bash -c "rm -rf /tmp/pgx-profile-data && /workspace/postgres-debug/bin/initdb -D /tmp/pgx-profile-data -U $(USER)"
	@docker exec pgx-lower-dev tar czf /workspace/pgx-profile-data.tar.gz -C /tmp pgx-profile-data
	@tar xzf ../pgx-profile-data.tar.gz -C $(PGX_PROFILE_DIR)
	@mv $(PGX_PROFILE_DIR)/pgx-profile-data $(PGX_PROFILE_DIR)/data
	@echo "port = 54325" >> $(PGX_PROFILE_DIR)/data/postgresql.conf
	@echo "Starting PostgreSQL..."
	@LD_LIBRARY_PATH=$(PGX_PROFILE_DIR)/lib/compat $(PGX_PROFILE_DIR)/bin/pg_ctl -D $(PGX_PROFILE_DIR)/data -l /tmp/pgx-profile.log start
	@sleep 2
	@psql -h localhost -p 54325 -U $(USER) -d postgres -c "CREATE USER postgres SUPERUSER;"
	@echo ""
	@echo "All containers ready! PostgreSQL started automatically."
	@echo "  Dev (54320):        psql -h localhost -p 54320 -U postgres"
	@echo "  Benchmark (54322):  psql -h localhost -p 54322 -U postgres"
	@echo "  Host-Profile (54325): psql -h localhost -p 54325 -U $(USER) -d postgres"

rebuild: clean
	docker compose build dev --no-cache
	docker compose up -d dev
	@./dev/build-ptest.sh

build-debug:
	@echo "Building Debug configuration in dev container..."
	@echo "Ensuring dev container is running with latest configuration..."
	@docker compose up -d --force-recreate --no-deps dev
	@sleep 2
	@echo "Ensuring PostgreSQL is running in dev container (port 54320)..."
	@docker exec pgx-lower-dev bash -c " \
		if ! pgrep -x postgres > /dev/null; then \
			echo 'Starting PostgreSQL...'; \
			su - postgres -c '/usr/local/pgsql/bin/pg_ctl -D /var/lib/postgresql/data -l /var/lib/postgresql/logfile start'; \
			sleep 2; \
		fi && \
		su - postgres -c '/usr/local/pgsql/bin/pg_ctl -D /var/lib/postgresql/data status' \
	"
	@docker exec pgx-lower-dev bash -c " \
		cd /workspace && \
		mkdir -p build-docker-ptest && \
		cd build-docker-ptest && \
		if [ ! -f CMakeCache.txt ]; then \
			echo 'Configuring Debug build...'; \
			cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug -DBUILD_ONLY_EXTENSION=ON .. || true; \
			cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug -DBUILD_ONLY_EXTENSION=ON ..; \
		fi && \
		echo 'Building...' && \
		cmake --build . && \
		echo 'Installing extension into dev PostgreSQL (/usr/local/pgsql/lib)...' && \
		cmake --install . \
	"
	@echo "Debug build completed: build-docker-ptest/extension/pgx_lower.so"
	@echo "PostgreSQL running on host port 54320"

ptest-debug: build-debug
	@echo "Running PostgreSQL regression tests (Debug build in dev container on port 54320)..."
	@docker exec pgx-lower-dev chown -R postgres:postgres /workspace/build-docker-ptest
	@docker exec -u postgres pgx-lower-dev bash -c " \
		cd /workspace/build-docker-ptest && \
		ctest --output-on-failure \
	" || (echo "Tests failed - check build-docker-ptest/extension/regression.diffs"; exit 1)
	@echo "Debug regression tests completed!"
	@echo "Connect to test database: psql -h localhost -p 54320 -U postgres"

build-release:
	@echo "Building Release configuration in dev container..."
	@echo "Ensuring dev container is running with latest configuration..."
	@docker compose up -d --force-recreate --no-deps dev
	@sleep 2
	@echo "Ensuring PostgreSQL is running in dev container (port 54320)..."
	@docker exec pgx-lower-dev bash -c " \
		if ! pgrep -x postgres > /dev/null; then \
			echo 'Starting PostgreSQL...'; \
			su - postgres -c '/usr/local/pgsql/bin/pg_ctl -D /var/lib/postgresql/data -l /var/lib/postgresql/logfile start'; \
			sleep 2; \
		fi && \
		su - postgres -c '/usr/local/pgsql/bin/pg_ctl -D /var/lib/postgresql/data status' \
	"
	@docker exec pgx-lower-dev bash -c " \
		cd /workspace && \
		mkdir -p build-docker-ptest-release && \
		cd build-docker-ptest-release && \
		if [ ! -f CMakeCache.txt ]; then \
			echo 'Configuring Release build (using -O2 -g to avoid MLIR bug)...'; \
			cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_ONLY_EXTENSION=ON .. || true; \
			cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_ONLY_EXTENSION=ON ..; \
		fi && \
		echo 'Building...' && \
		cmake --build . && \
		echo 'Installing extension into dev PostgreSQL...' && \
		cmake --install . \
	"
	@echo "Release build completed: build-docker-ptest-release/extension/pgx_lower.so"
	@echo "PostgreSQL running on host port 54320"

ptest-release: build-release
	@echo "Running PostgreSQL regression tests (Release build in dev container on port 54320)..."
	@docker exec pgx-lower-dev chown -R postgres:postgres /workspace/build-docker-ptest-release
	@docker exec -u postgres pgx-lower-dev bash -c " \
		cd /workspace/build-docker-ptest-release && \
		ctest --output-on-failure \
	" || (echo "Tests failed - check build-docker-ptest-release/extension/regression.diffs"; exit 1)
	@echo "Release regression tests completed!"
	@echo "Connect to test database: psql -h localhost -p 54320 -U postgres"

build-profile:
	@echo "Building Profile configuration (with -pg instrumentation) in dev container..."
	@echo "Ensuring dev container is running with latest configuration..."
	@docker compose up -d --force-recreate --no-deps dev
	@sleep 2
	@echo "Ensuring PostgreSQL is running in dev container (port 54320)..."
	@docker exec pgx-lower-dev bash -c " \
		if ! pgrep -x postgres > /dev/null; then \
			echo 'Starting PostgreSQL...'; \
			su - postgres -c '/usr/local/pgsql/bin/pg_ctl -D /var/lib/postgresql/data -l /var/lib/postgresql/logfile start'; \
			sleep 2; \
		fi && \
		su - postgres -c '/usr/local/pgsql/bin/pg_ctl -D /var/lib/postgresql/data status' \
	"
	@docker exec pgx-lower-dev bash -c " \
		cd /workspace && \
		mkdir -p build-docker-ptest-profile && \
		cd build-docker-ptest-profile && \
		if [ ! -f CMakeCache.txt ]; then \
			echo 'Configuring Profile build (-O2 -g -pg -fno-omit-frame-pointer)...'; \
			cmake -G Ninja -DCMAKE_BUILD_TYPE=Profile -DBUILD_ONLY_EXTENSION=ON .. || true; \
			cmake -G Ninja -DCMAKE_BUILD_TYPE=Profile -DBUILD_ONLY_EXTENSION=ON ..; \
		fi && \
		echo 'Building...' && \
		cmake --build . && \
		echo 'Installing extension into dev PostgreSQL...' && \
		cmake --install . \
	"
	@echo "Profile build completed: build-docker-ptest-profile/extension/pgx_lower.so"
	@echo "PostgreSQL running on host port 54320"

ptest-profile: build-profile
	@echo "Running PostgreSQL regression tests (Profile build in dev container on port 54320)..."
	@docker exec pgx-lower-dev chown -R postgres:postgres /workspace/build-docker-ptest-profile
	@docker exec -u postgres pgx-lower-dev bash -c " \
		cd /workspace/build-docker-ptest-profile && \
		ctest --output-on-failure \
	" || (echo "Tests failed - check build-docker-ptest-profile/extension/regression.diffs"; exit 1)
	@echo "Profile regression tests completed!"
	@echo "Connect to test database: psql -h localhost -p 54320 -U postgres"

release-container: build-release
	@echo "Deploying release build to benchmark container..."
	@echo "Ensuring benchmark container is running with latest configuration..."
	@docker compose up -d --force-recreate --no-deps benchmark
	@sleep 3
	@echo "Installing extension into benchmark container..."
	@docker exec pgx-lower-benchmark bash -c " \
		mkdir -p /usr/local/pgsql/lib /usr/local/pgsql/share/extension && \
		cp -v /workspace/build-docker-ptest-release/extension/pgx_lower.so /usr/local/pgsql/lib/ && \
		cp -v /workspace/extension/control/pgx_lower.control /usr/local/pgsql/share/extension/ && \
		cp -v /workspace/extension/sql/pgx_lower--*.sql /usr/local/pgsql/share/extension/ 2>/dev/null || true && \
		ls -lh /usr/local/pgsql/lib/pgx_lower.so /usr/local/pgsql/share/extension/pgx_lower.control \
	"
	@echo "Initializing database in benchmark container..."
	@docker exec -u postgres pgx-lower-benchmark bash -c " \
		if [ ! -f /var/lib/postgresql/data/PG_VERSION ]; then \
			/usr/local/pgsql/bin/initdb -D /var/lib/postgresql/data; \
		else \
			echo 'Database already initialized'; \
		fi \
	"
	@echo "Benchmark container ready!"
	@echo "PostgreSQL in benchmark container (host port 54322)"
	@echo "Start PostgreSQL: docker exec -u postgres pgx-lower-benchmark /usr/local/pgsql/bin/postgres -D /var/lib/postgresql/data"
	@echo "Connect: psql -h localhost -p 54322 -U postgres"

deploy-profile-extension:
	@echo "Deploying release extension to profile container..."
	@docker exec pgx-lower-profile bash -c " \
		mkdir -p /usr/local/pgsql/lib /usr/local/pgsql/share/extension && \
		cp -v /workspace/build-docker-ptest-release/extension/pgx_lower.so /usr/local/pgsql/lib/ && \
		cp -v /workspace/extension/control/pgx_lower.control /usr/local/pgsql/share/extension/ && \
		cp -v /workspace/extension/sql/pgx_lower--*.sql /usr/local/pgsql/share/extension/ 2>/dev/null || true \
	"
	@echo "Profile extension deployed!"

# PostgreSQL debug build for profiling
build-postgres-debug:
	@echo "Building PostgreSQL with debug symbols in dev container..."
	@echo "Ensuring dev container is running..."
	@docker compose up -d --no-deps dev
	@sleep 2
	@docker exec pgx-lower-dev bash -c " \
		cd /workspace && \
		if [ -d postgres-debug/bin ]; then \
			echo 'postgres-debug already exists (has bin directory)'; \
			echo 'Delete /workspace/postgres-debug to rebuild'; \
			exit 0; \
		fi && \
		echo 'Downloading PostgreSQL 17.6...' && \
		wget -q https://ftp.postgresql.org/pub/source/v17.6/postgresql-17.6.tar.gz && \
		echo 'Extracting...' && \
		tar -xzf postgresql-17.6.tar.gz && \
		cd postgresql-17.6 && \
		echo 'Configuring with debug symbols (CC=clang CFLAGS=\"-g -O2\")...' && \
		CC=clang CXX=clang++ CFLAGS='-O2 -g -pg -fno-omit-frame-pointer' LDFLAGS='-pg' ./configure --prefix=/workspace/postgres-debug --with-openssl --with-icu && \
		echo 'Building PostgreSQL (this takes 5-10 minutes)...' && \
		make -j\$$(nproc) && \
		echo 'Installing to /workspace/postgres-debug...' && \
		make install && \
		cd .. && \
		rm -rf postgresql-17.6 postgresql-17.6.tar.gz && \
		echo 'PostgreSQL debug build completed!' \
	"
	@echo "PostgreSQL debug build available at: /workspace/postgres-debug"

build-perf:
	@echo "Building perf from kernel sources in dev container..."
	@echo "Ensuring dev container is running..."
	@docker compose up -d --no-deps dev
	@sleep 2
	@docker exec pgx-lower-dev bash -c " \
		cd /workspace/docker && \
		if [ -f perf-built ]; then \
			echo 'perf already built (perf-built file exists)'; \
			echo 'Delete /workspace/docker/perf-built to rebuild'; \
			exit 0; \
		fi && \
		KERNEL_VERSION=\$$(uname -r | sed 's/-generic//') && \
		echo \"Downloading kernel source for \$$KERNEL_VERSION...\" && \
		wget -q https://cdn.kernel.org/pub/linux/kernel/v6.x/linux-\$$KERNEL_VERSION.tar.xz && \
		echo 'Extracting...' && \
		tar -xf linux-\$$KERNEL_VERSION.tar.xz && \
		cd linux-\$$KERNEL_VERSION/tools/perf && \
		echo 'Building perf...' && \
		make -j\$$(nproc) && \
		echo 'Copying perf binary to /workspace/docker/perf-built...' && \
		cp perf /workspace/docker/perf-built && \
		cd /workspace/docker && \
		rm -rf linux-\$$KERNEL_VERSION linux-\$$KERNEL_VERSION.tar.xz && \
		echo 'perf build completed!' \
	"
	@echo "perf binary available at: /workspace/docker/perf-built"

# Host installation for profiling (no Docker overhead)
PGX_PROFILE_DIR ?= $(HOME)/.pgx-profile

install-host-profile: build-profile build-postgres-debug
	@echo "Installing PostgreSQL + pgx_lower (with -pg profiling) to $(PGX_PROFILE_DIR)..."
	@rm -rf $(PGX_PROFILE_DIR)
	@mkdir -p $(PGX_PROFILE_DIR)
	@echo "Extracting postgres-debug (with -pg) from container..."
	@docker exec pgx-lower-dev tar czf /workspace/postgres-debug.tar.gz -C /workspace postgres-debug
	@docker exec pgx-lower-dev bash -c "cd /usr/lib/x86_64-linux-gnu && tar czf /workspace/icu74.tar.gz libicu*.so.74*"
	@docker exec pgx-lower-dev bash -c "cd /usr/lib/llvm-20/lib && tar czf /workspace/mlir-libs.tar.gz libMLIR*.so* libLLVM*.so*"
	@tar xzf ../postgres-debug.tar.gz -C $(PGX_PROFILE_DIR) --strip-components=1
	@mkdir -p $(PGX_PROFILE_DIR)/lib/compat
	@tar xzf ../icu74.tar.gz -C $(PGX_PROFILE_DIR)/lib/compat
	@tar xzf ../mlir-libs.tar.gz -C $(PGX_PROFILE_DIR)/lib/compat
	@echo "Installing pgx_lower extension (with -pg profiling)..."
	@cp ../build-docker-ptest-profile/extension/pgx_lower.so $(PGX_PROFILE_DIR)/lib/
	@cp ../extension/control/pgx_lower.control $(PGX_PROFILE_DIR)/share/extension/
	@cp ../extension/sql/pgx_lower--1.0.sql $(PGX_PROFILE_DIR)/share/extension/
	@echo ""
	@echo "PostgreSQL + pgx_lower with -pg instrumentation installed successfully!"
	@echo "Database will be initialized when running benchmarks."

# Profiling container targets
profile-container: build-postgres-debug build-release
	@echo "Deploying to profile container..."
	@echo "Ensuring profile container is running with latest configuration..."
	@docker compose up -d --force-recreate --no-deps profile
	@sleep 3
	@echo "Installing PostgreSQL debug build into profile container..."
	@docker exec pgx-lower-profile bash -c " \
		if [ ! -d /workspace/postgres-debug/bin ]; then \
			echo 'ERROR: /workspace/postgres-debug not found!'; \
			echo 'Run: cd docker && make build-postgres-debug'; \
			exit 1; \
		fi && \
		rm -rf /usr/local/pgsql && \
		cp -r /workspace/postgres-debug /usr/local/pgsql && \
		ls -lh /usr/local/pgsql/bin/postgres \
	"
	@echo "Installing pgx_lower extension into profile container..."
	@docker exec pgx-lower-profile bash -c " \
		mkdir -p /usr/local/pgsql/lib /usr/local/pgsql/share/extension && \
		cp -v /workspace/build-docker-ptest-release/extension/pgx_lower.so /usr/local/pgsql/lib/ && \
		cp -v /workspace/extension/control/pgx_lower.control /usr/local/pgsql/share/extension/ && \
		cp -v /workspace/extension/sql/pgx_lower--*.sql /usr/local/pgsql/share/extension/ 2>/dev/null || true && \
		ls -lh /usr/local/pgsql/lib/pgx_lower.so /usr/local/pgsql/share/extension/pgx_lower.control \
	"
	@echo "Initializing database in profile container..."
	@docker exec -u postgres pgx-lower-profile bash -c " \
		if [ ! -f /var/lib/postgresql/data/PG_VERSION ]; then \
			/usr/local/pgsql/bin/initdb -D /var/lib/postgresql/data; \
		else \
			echo 'Database already initialized'; \
		fi \
	"
	@echo "Profile container ready!"
	@echo "PostgreSQL in profile container (host port 54321)"
	@echo "Start PostgreSQL: docker exec -u postgres pgx-lower-profile /usr/local/pgsql/bin/postgres -D /var/lib/postgresql/data"
	@echo "Connect: psql -h localhost -p 54321 -U postgres"

build-benchmark:
	@echo "Building pgx-lower benchmark container..."
	@echo "Step 1: Building release extension in dev container"
	@./dev/build-release.sh
	@echo "Step 2: Removing old benchmark volume to ensure fresh .so"
	docker compose down benchmark
	docker volume rm docker_benchmark-data 2>/dev/null || true
	@echo "Step 3: Creating slim benchmark container with release .so"
	docker compose build benchmark
	@echo "Step 4: Starting benchmark container"
	docker compose up -d benchmark

up-benchmark:
	docker compose up -d benchmark

down-benchmark:
	docker compose stop benchmark
	docker compose rm -f benchmark

restart-benchmark:
	docker compose restart benchmark

shell-benchmark:
	docker exec -it pgx-lower-benchmark bash

logs-benchmark:
	docker logs -f pgx-lower-benchmark

clean-benchmark:
	docker compose down benchmark -v
	docker rmi pgx-lower:benchmark || true

# Benchmark orchestration targets

BENCH_PROFILE ?= quick

bench-dockers:
	@echo "Benchmark Profile: $(BENCH_PROFILE)"
	@echo "Step 1: Installing Python dependencies..."
	@python3 -c "import yaml, psycopg2, psutil, lz4.frame" 2>/dev/null || \
		sudo apt-get install -y python3-yaml python3-psycopg2 python3-psutil python3-lz4
	@echo "Step 2: Ensuring all containers are running..."
	@$(MAKE) all
	@echo "Step 3: Running benchmark profile '$(BENCH_PROFILE)'..."
	@python3 ../benchmark/run_benchmark_config.py $(BENCH_PROFILE) --config ../benchmark-config.yaml
	@echo ""
	@echo "Benchmarks complete! Results stored in benchmark/output/benchmark.db"

bench-list:
	@python3 ../benchmark/run_benchmark_config.py --list --config ../benchmark-config.yaml
