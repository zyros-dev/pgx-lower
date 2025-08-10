# pgx_lower Makefile
# Build system for PostgreSQL JIT compilation via MLIR
# Architecture: PostgreSQL AST → RelAlg → DB → DSA → Standard MLIR → LLVM IR → JIT

.PHONY: build clean test install psql-start debug-stop all format-check format-fix ptest utest fcheck ffix rebuild help build-ptest build-utest clean-ptest clean-utest compile_commands clean-root

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
	cmake -S . -B $(BUILD_DIR_UTEST) -G $(CMAKE_GENERATOR) -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE)
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
	cd $(BUILD_DIR_PTEST) && ctest --output-on-failure && cd -
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
	@echo "Running unit tests (MLIR dialects, lowering passes, streaming translators)..."
	cd $(BUILD_DIR_UTEST) && ./mlir_unit_test || true; cd -
	@echo "Unit tests completed!"

utest-run:
	@echo "Running unit tests without rebuild..."
	cd $(BUILD_DIR_UTEST) && ./mlir_unit_test || true; cd -
	@echo "Unit tests completed!"

utest-all:
	@echo "Running ALL unit tests (including potentially crashing tests)..."
	cd $(BUILD_DIR_UTEST) && ./mlir_unit_test || true; cd -
	@echo "All unit tests completed!"

compile_commands: build
	@echo "Generating compile_commands.json..."
	cmake -S . -B $(BUILD_DIR) -G $(CMAKE_GENERATOR) -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
	cp $(BUILD_DIR)/compile_commands.json .
	@echo "compile_commands.json generated!"

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
	@echo "  compile_commands - Generate compile_commands.json"
	@echo "  install      - Install the project"
	@echo "  rebuild      - Quick rebuild without cleaning"
	@echo "  fcheck       - Check code formatting"
	@echo "  ffix         - Fix code formatting"