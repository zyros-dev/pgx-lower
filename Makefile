# pgx_lower Makefile
# Build system for PostgreSQL JIT compilation via MLIR
# Architecture: PostgreSQL AST → RelAlg → DB → DSA → Standard MLIR → LLVM IR → JIT

.PHONY: build clean test install psql-start debug-stop all format-check format-fix ptest utest fcheck ffix rebuild help build-ptest build-utest clean-ptest clean-utest compile_commands clean-root gviz bench psql-bench lingo-bench validate-bench venv

# Build directories for different test types
BUILD_DIR = build
BUILD_DIR_PTEST = build-ptest
BUILD_DIR_UTEST = build-utest
CMAKE_GENERATOR = Ninja
CMAKE_BUILD_TYPE = Debug

all: test

build:
	@echo "Building project..."
	cmake -S . -B $(BUILD_DIR) -G $(CMAKE_GENERATOR) -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE)
	cmake --build $(BUILD_DIR)
	@echo "Build completed!"

build-ptest:
	@echo "Building PostgreSQL extension for regression tests..."
	@echo "Pipeline: AST → RelAlg → DB → DSA → Standard MLIR → LLVM"
	cmake -S . -B $(BUILD_DIR_PTEST) -G $(CMAKE_GENERATOR) -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) -DBUILD_ONLY_EXTENSION=ON
	cmake --build $(BUILD_DIR_PTEST)
	@echo "PostgreSQL extension build completed!"

build-utest:
	@echo "Building project for unit tests..."
	cmake -S . -B $(BUILD_DIR_UTEST) -G $(CMAKE_GENERATOR) -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) -DBUILDING_UNIT_TESTS=ON
	cmake --build $(BUILD_DIR_UTEST)
	@echo "Unit test build completed!"

clean: clean-root
	@echo "Cleaning all build directories..."
	rm -rf $(BUILD_DIR) $(BUILD_DIR_PTEST) $(BUILD_DIR_UTEST)
	@echo "Clean completed!"

clean-root:
	@echo "Cleaning CMake files from root directory..."
	rm -rf CMakeFiles/ CMakeCache.txt cmake_install.cmake *.log
	rm -rf _deps/ Testing/ CTestTestfile.cmake
	find . -maxdepth 1 -name "*.ninja*" -delete
	find . -maxdepth 1 -name "*.cmake" -not -name "CMakeLists.txt" -delete
	@echo "Root directory cleaned!"

clean-ptest:
	@echo "Cleaning PostgreSQL test build directory..."
	rm -rf $(BUILD_DIR_PTEST)
	@echo "PostgreSQL test clean completed!"

clean-utest:
	@echo "Cleaning unit test build directory..."
	rm -rf $(BUILD_DIR_UTEST)
	@echo "Unit test clean completed!"

install: build
	@echo "Installing..."
	sudo cmake --install $(BUILD_DIR)
	@echo "Install completed!"

install-ptest: build-ptest
	@echo "Installing PostgreSQL test build..."
	sudo cmake --install $(BUILD_DIR_PTEST)
	@echo "PostgreSQL test install completed!"

ptest: install-ptest
	@echo "Running PostgreSQL regression tests (Test 1: SELECT * FROM test)..."
	-cd $(BUILD_DIR_PTEST) && ctest --output-on-failure && cd -
	@if [ -f .venv/bin/python3 ]; then \
		.venv/bin/python3 tools/validate_tpch.py $(BUILD_DIR_PTEST)/extension/results/tpch.out $(BUILD_DIR_PTEST)/extension/results/tpch_no_lower.out; \
	else \
		echo "Warning: .venv not found, using system python3"; \
		python3 tools/validate_tpch.py $(BUILD_DIR_PTEST)/extension/results/tpch.out $(BUILD_DIR_PTEST)/extension/results/tpch_no_lower.out; \
	fi
	@echo "PostgreSQL regression tests completed!"

psql-start:
	@echo "Starting PostgreSQL for debugging..."
	sudo -u postgres LD_LIBRARY_PATH=/usr/lib/llvm-20/lib /usr/local/pgsql/bin/postgres -D /usr/local/pgsql/data

rebuild:
	@echo "Rebuilding..."
	cmake --build $(BUILD_DIR)
	@echo "Rebuild completed!"

rebuild-ptest:
	@echo "Rebuilding PostgreSQL test build..."
	cmake --build $(BUILD_DIR_PTEST)
	@echo "PostgreSQL test rebuild completed!"

rebuild-utest:
	@echo "Rebuilding unit test build..."
	cmake --build $(BUILD_DIR_UTEST)
	@echo "Unit test rebuild completed!"

utest: build-utest
	@echo "Running unit tests..."
	cd $(BUILD_DIR_UTEST) && ctest --output-on-failure --verbose
	@echo "Unit tests completed!"

utest-run:
	@echo "Running unit tests without rebuild..."
	cd $(BUILD_DIR_UTEST) && ctest --output-on-failure --verbose
	@echo "Unit tests completed!"

utest-all:
	@echo "Running ALL unit tests (including potentially crashing tests)..."
	cd $(BUILD_DIR_UTEST) && ./tests/unit/mlir/test_standard_to_llvm_pass; cd -
	@echo "All unit tests completed!"

compile_commands:
	@echo "Generating compile_commands.json..."
	@echo "Using PostgreSQL extension build (ptest) which is known to work..."
	cmake -S . -B $(BUILD_DIR_PTEST) -G $(CMAKE_GENERATOR) -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) -DBUILD_ONLY_EXTENSION=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
	@if [ -f $(BUILD_DIR_PTEST)/compile_commands.json ]; then \
		cp $(BUILD_DIR_PTEST)/compile_commands.json .; \
		echo "compile_commands.json generated from PostgreSQL extension build!"; \
		echo "This includes all core source files used by the extension."; \
	else \
		echo "Warning: compile_commands.json not found in ptest build, trying full build..."; \
		cmake -S . -B $(BUILD_DIR) -G $(CMAKE_GENERATOR) -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) -DCMAKE_EXPORT_COMPILE_COMMANDS=ON; \
		if [ -f $(BUILD_DIR)/compile_commands.json ]; then \
			cp $(BUILD_DIR)/compile_commands.json .; \
			echo "compile_commands.json generated from full build!"; \
		else \
			echo "Error: Could not generate compile_commands.json"; \
		fi; \
	fi

fcheck:
	@echo "Checking clang-format errors in source code..."
	@find src/ -name "*.cpp" -o -name "*.c" -o -name "*.h" | while read file; do \
		if ! clang-format --dry-run --Werror "$$file" >/dev/null 2>&1; then \
			echo "  $$file"; \
			clang-format --dry-run --Werror "$$file" || true; \
		fi; \
	done
	@echo "Format check completed!"

ffix:
	@echo "Fixing clang-format errors in source code..."
	@find src/ -name "*.cpp" -o -name "*.c" -o -name "*.h" | xargs clang-format -i
	@echo "Format fix completed!"

gviz:
	@echo "Generating CMake dependency visualization..."
	@echo "Creating build-viz directory and generating graphviz files..."
	@mkdir -p build-viz
	@cd build-viz && cmake --graphviz=pgx_lower_deps.dot .. >/dev/null 2>&1
	@echo "Converting to SVG format..."
	@cd build-viz && dot -Tsvg pgx_lower_deps.dot -o pgx_lower_deps.svg
	@echo "Creating clean version (filtering out Google Test dependencies)..."
	@cd build-viz && grep -v "googletest\|gmock\|gtest" pgx_lower_deps.dot > clean_deps.dot
	@cd build-viz && dot -Tsvg clean_deps.dot -o clean_pgx_lower_deps.svg
	@echo "Generated visualizations:"
	@echo "  - build-viz/pgx_lower_deps.svg (complete dependency graph)"
	@echo "  - build-viz/clean_pgx_lower_deps.svg (project dependencies only)"
	@echo "Opening visualization..."
	@if which firefox >/dev/null 2>&1; then \
		firefox build-viz/clean_pgx_lower_deps.svg >/dev/null 2>&1 & \
	else \
		echo "Install firefox to auto-open SVG files"; \
	fi
	@echo "CMake visualization completed!"

venv:
	@if [ ! -d .venv ]; then \
		echo "Creating Python 3.12.11 virtual environment..."; \
		if [ ! -d ~/.pyenv/versions/3.12.11 ]; then \
			echo "Python 3.12.11 not installed. Run: ~/.pyenv/bin/pyenv install 3.12.11"; \
			exit 1; \
		fi; \
		~/.pyenv/versions/3.12.11/bin/python3 -m venv .venv; \
		~/.pyenv/versions/3.12.11/bin/python3 -c "import sys; print('.venv created with Python', sys.version.split()[0])"; \
		echo "Activate with: source .venv/bin/activate.fish (or .venv/bin/activate for bash)"; \
	else \
		echo "Virtual environment already exists at .venv"; \
	fi

pgx-bench:
	@SF_VALUE=$${SF:-0.1}; \
	echo "Running TPC-H benchmark (pgx-lower) with scale factor $$SF_VALUE..."; \
	python3 tools/bench.py pgx $$SF_VALUE

psql-bench:
	@SF_VALUE=$${SF:-0.1}; \
	echo "Running TPC-H benchmark (vanilla PostgreSQL) with scale factor $$SF_VALUE..."; \
	python3 tools/bench.py psql $$SF_VALUE

lingo-bench:
	@SF_VALUE=$${SF:-0.1}; \
	echo "Running TPC-H benchmark (LingoDB) with scale factor $$SF_VALUE..."; \
	python3 tools/bench.py lingo $$SF_VALUE

validate-bench:
	@SF_VALUE=$${SF:-0.1}; \
	echo "Running TPC-H validation benchmark (all engines) with scale factor $$SF_VALUE..."; \
	python3 tools/bench.py validate $$SF_VALUE

help:
	@echo "Available targets:"
	@echo "  build        - Build the main project"
	@echo "  build-ptest  - Build for PostgreSQL tests"
	@echo "  build-utest  - Build for unit tests"
	@echo "  clean        - Clean all build directories and root"
	@echo "  clean-root   - Clean CMake files from root directory"
	@echo "  ptest        - Run PostgreSQL regression tests"
	@echo "  utest        - Build and run unit tests (excludes crashing tests)"
	@echo "  utest-run    - Run unit tests without rebuild (excludes crashing tests)"
	@echo "  utest-all    - Run ALL unit tests including potentially crashing ones"
	@echo "  venv         - Create Python virtual environment (Python 3.12.11)"
	@echo "  bench        - Run TPC-H benchmark with pgx-lower (Usage: make bench [SF=<scale_factor>], default SF=0.1)"
	@echo "  psql-bench   - Run TPC-H benchmark with vanilla PostgreSQL (Usage: make psql-bench [SF=<scale_factor>])"
	@echo "  lingo-bench  - Run TPC-H benchmark with LingoDB (Usage: make lingo-bench [SF=<scale_factor>])"
	@echo "  validate-bench - Cross-validate all benchmark engines (Usage: make validate-bench [SF=<scale_factor>])"
	@echo "  compile_commands - Generate compile_commands.json"
	@echo "  install      - Install the project"
	@echo "  rebuild      - Quick rebuild without cleaning"
	@echo "  fcheck       - Check code formatting"
	@echo "  ffix         - Fix code formatting"
	@echo "  gviz         - Generate CMake dependency visualization graphs"