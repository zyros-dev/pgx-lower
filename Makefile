# pgx_lower Makefile

.PHONY: build clean test install psql-start debug-stop all format-check format-fix ptest utest fcheck ffix rebuild help

BUILD_DIR = build
CMAKE_GENERATOR = Ninja
CMAKE_BUILD_TYPE = Debug

all: test

build:
	@echo "Building project..."
	cmake -S . -B $(BUILD_DIR) -G $(CMAKE_GENERATOR) -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE)
	cmake --build $(BUILD_DIR)
	@echo "Build completed!"

clean:
	@echo "Cleaning build directory..."
	rm -rf $(BUILD_DIR)
	@echo "Clean completed!"


install: build
	@echo "Installing..."
	sudo cmake --install $(BUILD_DIR)
	@echo "Install completed!"

ptest: install
	@echo "Building, installing, and testing..."
	cmake -S . -B $(BUILD_DIR) -G $(CMAKE_GENERATOR) -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE)
	sudo cmake --build $(BUILD_DIR)
	sudo cmake --install $(BUILD_DIR)
	cd $(BUILD_DIR) && ctest --output-on-failure && cd -
	@echo "Tests completed!"

psql-start:
	@echo "Starting PostgreSQL for debugging..."
	sudo -u postgres LD_LIBRARY_PATH=/usr/lib/llvm-20/lib /usr/local/pgsql/bin/postgres -D /usr/local/pgsql/data

rebuild:
	@echo "Rebuilding..."
	cmake --build $(BUILD_DIR)
	@echo "Rebuild completed!"

utest: build
	@echo "Running unit tests..."
	cd $(BUILD_DIR) && ./mlir_unit_test && cd -
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