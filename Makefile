# pgx_lower Makefile

.PHONY: help build clean test install debug-start debug-stop all

BUILD_DIR = build
CMAKE_GENERATOR = Ninja
CMAKE_BUILD_TYPE = Debug

all: build

help:
	@echo "Available targets:"
	@echo "  build        - Build the project"
	@echo "  clean        - Clean build directory"
	@echo "  test         - Build, install, and run tests"
	@echo "  install      - Build and install the extension"
	@echo "  debug-start  - Start PostgreSQL for debugging"
	@echo "  debug-stop   - Stop debugging processes"
	@echo "  all          - Build the project (default)"

build:
	@echo "Building project..."
	rm -rf $(BUILD_DIR)
	cmake -S . -B $(BUILD_DIR) -G $(CMAKE_GENERATOR) -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE)
	cmake --build $(BUILD_DIR)
	@echo "Build completed!"

clean:
	@echo "Cleaning build directory..."
	rm -rf $(BUILD_DIR)
	@echo "Clean completed!"

test: clean
	@echo "Building, installing, and testing..."
	cmake -S . -B $(BUILD_DIR) -G $(CMAKE_GENERATOR) -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE)
	sudo cmake --build $(BUILD_DIR)
	sudo cmake --install $(BUILD_DIR)
	cd $(BUILD_DIR) && ctest --output-on-failure && cd -
	@echo "Tests completed!"

install: clean
	@echo "Building and installing..."
	cmake -S . -B $(BUILD_DIR) -G $(CMAKE_GENERATOR) -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE)
	sudo cmake --build $(BUILD_DIR)
	sudo cmake --install $(BUILD_DIR)
	@echo "Install completed!"

psql-start:
	@echo "Starting PostgreSQL for debugging..."
	sudo -u postgres LD_LIBRARY_PATH=/usr/lib/llvm-20/lib /usr/local/pgsql/bin/postgres -D /usr/local/pgsql/data

debug-stop:
	@echo "Stopping debugging processes..."
	ps aux | grep '[g]dbserver' | awk '{print $$2}' | xargs -r kill -9
	@echo "Debugging processes stopped"

rebuild:
	@echo "Rebuilding..."
	cmake --build $(BUILD_DIR)
	@echo "Rebuild completed!"

unit-test: build
	@echo "Running unit tests..."
	cd $(BUILD_DIR) && ctest --output-on-failure -R mlir_unit_test && cd -
	@echo "Unit tests completed!" 