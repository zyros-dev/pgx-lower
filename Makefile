# pgx_lower Makefile

.PHONY: build clean test install psql-start debug-stop all format-check format-fix ptest utest fcheck ffix rebuild help build-ptest build-utest clean-ptest clean-utest

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
	@echo "Building project for PostgreSQL tests..."
	cmake -S . -B $(BUILD_DIR_PTEST) -G $(CMAKE_GENERATOR) -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE)
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