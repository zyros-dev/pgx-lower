# pgx_lower Makefile

.PHONY: build clean test install psql-start debug-stop all format-check format-fix ptest utest utest-run fcheck ffix rebuild help build-ptest build-utest clean-ptest clean-utest compile_commands coverage

# Build directories for different test types
BUILD_DIR = build
BUILD_DIR_PTEST = build-ptest
BUILD_DIR_UTEST = build-utest
CMAKE_GENERATOR = Ninja
CMAKE_BUILD_TYPE = Debug
PY = /home/xzel/pyautogui-env/bin/python

all: test

build:
	@echo "Building project..."
	cmake -S . -B $(BUILD_DIR) -G $(CMAKE_GENERATOR) -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE)
	cmake --build $(BUILD_DIR)
	@echo "Build completed!"

build-ptest:
	@echo "Building project for PostgreSQL tests..."
	cmake -S . -B $(BUILD_DIR_PTEST) -G $(CMAKE_GENERATOR) -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) -DBUILD_ONLY_EXTENSION=ON
	cmake --build $(BUILD_DIR_PTEST)
	@echo "PostgreSQL test build completed!"

build-utest:
	@echo "Building project for unit tests..."
	cmake -S . -B $(BUILD_DIR_UTEST) -G $(CMAKE_GENERATOR) -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE)
	cmake --build $(BUILD_DIR_UTEST)
	@echo "Unit test build completed!"

clean:
	@echo "Cleaning all build directories..."
	rm -rf $(BUILD_DIR) $(BUILD_DIR_PTEST) $(BUILD_DIR_UTEST)
	@echo "Clean completed!"

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
	@echo "Running PostgreSQL tests..."
	cd $(BUILD_DIR_PTEST) && ctest --output-on-failure && cd -
	@echo "PostgreSQL tests completed!"

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
	cd $(BUILD_DIR_UTEST) && ./mlir_unit_test && cd -
	@echo "Unit tests completed!"

utest-run:
	@echo "Running unit tests without rebuild..."
	cd $(BUILD_DIR_UTEST) && ./mlir_unit_test && cd -
	@echo "Unit tests completed!"

fcheck:
	@echo "Checking clang-format and clang-tidy errors in source code..."
	@echo "=== Format Check ==="
	@find src/ -name "*.cpp" -o -name "*.c" -o -name "*.h" | xargs clang-format --dry-run --Werror || true
	@echo "=== Tidy Check (Parallel) ==="
	run-clang-tidy -j 8 -header-filter='.*' || true
	@echo "Format and tidy check completed!"

ffix:
	@echo "Fixing clang-format and clang-tidy errors in source code..."
	@echo "=== Format Fix ==="
	@find src/ -name "*.cpp" -o -name "*.c" -o -name "*.h" | xargs clang-format -i
	@echo "=== Tidy Fix (Parallel) ==="
	run-clang-tidy -j 8 -fix -fix-errors -header-filter='.*' || true
	@echo "Format and tidy fix completed!"

compile_commands:
	@echo "Generating compile_commands.json..."
	cmake -S . -B $(BUILD_DIR_PTEST) -G $(CMAKE_GENERATOR) -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) -DBUILD_ONLY_EXTENSION=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
	@ln -sf $(BUILD_DIR_PTEST)/compile_commands.json compile_commands.json
	@echo "compile_commands.json generated and symlinked to project root."

docs-server:
	$(PY) search_server.py

query_docs:
	$(PY) search_embeddings_cli.py $(QUERY)

coverage:
	@echo "Running unit tests with coverage..."
	@echo "Building with coverage enabled..."
	cmake -S . -B build-coverage -G $(CMAKE_GENERATOR) -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) -DENABLE_COVERAGE=ON
	cmake --build build-coverage
	@echo "Running unit tests..."
	cd build-coverage && ./mlir_unit_test && cd -
	@echo "Generating coverage report..."
	@if find build-coverage -name "*.gcda" -o -name "*.gcno" | grep -q .; then \
		echo "Using GCC coverage..."; \
		lcov --capture --directory build-coverage --output-file coverage.info 2>/dev/null || true; \
		lcov --remove coverage.info '*/build-coverage/*' '*/third_party/*' '*/usr/*' '*gtest*' '*gmock*' --output-file coverage_filtered.info 2>/dev/null || true; \
		mkdir -p coverage_report; \
		genhtml coverage_filtered.info --output-directory coverage_report 2>/dev/null || true; \
		echo "Coverage report generated in coverage_report/"; \
		echo "Open coverage_report/index.html in a browser to view the report"; \
		lcov --summary coverage_filtered.info 2>/dev/null || echo "Coverage summary not available"; \
		echo ""; \
		echo "=== CORE COMPONENT COVERAGE ==="; \
		cd build-coverage && gcov CMakeFiles/mlir_unit_test.dir/src/core/*.gcno 2>/dev/null | grep -E "File.*pgx-lower.*\.cpp|Lines executed" | grep -A1 "pgx-lower.*\.cpp" | sed 's|.*/||' && cd - >/dev/null; \
		echo ""; \
		echo "=== OVERALL COVERAGE ESTIMATE ==="; \
		cd build-coverage && gcov CMakeFiles/mlir_unit_test.dir/src/core/*.gcno 2>/dev/null | grep "Lines executed" | grep -v "0.00%" | awk -F'[%: ]' '{sum += $$3; count++} END {if(count > 0) printf "Average coverage: %.1f%%\n", sum/count; else print "No coverage data"}' && cd - >/dev/null; \
	else \
		echo "No coverage data found!"; \
	fi
	@echo "Coverage analysis completed!"