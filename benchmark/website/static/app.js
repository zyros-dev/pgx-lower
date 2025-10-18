/**
 * pgx-lower Benchmark Dashboard
 * Main application logic for chart rendering and data fetching
 */

// ============================================================================
// Global State
// ============================================================================

const state = {
    currentRun: 'latest',
    currentScale: null,
    autoRefresh: false,
    autoRefreshInterval: null,
    charts: {}, // Store Chart.js instances
    data: {
        overview: null,
        queries: null,
        aggregates: null,
        runs: null,
    }
};

// ============================================================================
// API Client
// ============================================================================

const api = {
    async get(endpoint) {
        try {
            const response = await fetch(endpoint);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            return await response.json();
        } catch (error) {
            console.error(`API error (${endpoint}):`, error);
            throw error;
        }
    },

    getOverview(scaleFactor, runTimestamp) {
        const params = new URLSearchParams();
        if (scaleFactor) params.append('scale_factor', scaleFactor);
        if (runTimestamp && runTimestamp !== 'latest') params.append('run_timestamp', runTimestamp);
        return this.get(`/api/overview?${params}`);
    },

    getQueries(pgxEnabled, scaleFactor, runTimestamp) {
        const params = new URLSearchParams();
        if (pgxEnabled !== null) params.append('pgx_enabled', pgxEnabled);
        if (scaleFactor) params.append('scale_factor', scaleFactor);
        if (runTimestamp && runTimestamp !== 'latest') params.append('run_timestamp', runTimestamp);
        return this.get(`/api/queries?${params}`);
    },

    getAggregates(scaleFactor) {
        const params = new URLSearchParams();
        if (scaleFactor) params.append('scale_factor', scaleFactor);
        return this.get(`/api/aggregates?${params}`);
    },

    getRuns() {
        return this.get('/api/runs');
    },

    getTimeseries(query, metric) {
        const params = new URLSearchParams({ query, metric });
        return this.get(`/api/timeseries?${params}`);
    },

    getHeatmap(scaleFactor, runTimestamp) {
        const params = new URLSearchParams();
        if (scaleFactor) params.append('scale_factor', scaleFactor);
        if (runTimestamp && runTimestamp !== 'latest') params.append('run_timestamp', runTimestamp);
        return this.get(`/api/heatmap?${params}`);
    },

    getProfileList(scaleFactor, runTimestamp) {
        const params = new URLSearchParams();
        if (scaleFactor) params.append('scale_factor', scaleFactor);
        if (runTimestamp && runTimestamp !== 'latest') params.append('run_timestamp', runTimestamp);
        return this.get(`/api/profile/list?${params}`);
    },

    getProfileStats() {
        return this.get('/api/profile/stats');
    }
};

// ============================================================================
// Initialization
// ============================================================================

document.addEventListener('DOMContentLoaded', async () => {
    console.log('Initializing dashboard...');

    // Load initial data
    await loadRuns();
    await loadData();

    // Setup event listeners
    setupEventListeners();

    // Render dashboard
    renderDashboard();

    console.log('Dashboard ready!');
});

async function loadRuns() {
    try {
        state.data.runs = await api.getRuns();

        // Populate run selector
        const runSelect = document.getElementById('run-select');
        runSelect.innerHTML = '<option value="latest">Latest</option>';

        state.data.runs.forEach(run => {
            const option = document.createElement('option');
            option.value = run.run_timestamp;
            option.textContent = `${run.run_timestamp} (${run.query_count} queries, SF=${run.scale_factor})`;
            runSelect.appendChild(option);
        });

        // Populate scale selector
        const scales = [...new Set(state.data.runs.map(r => r.scale_factor))];
        const scaleSelect = document.getElementById('scale-select');
        scaleSelect.innerHTML = '';

        scales.forEach(scale => {
            const option = document.createElement('option');
            option.value = scale;
            option.textContent = scale;
            scaleSelect.appendChild(option);
        });

        if (scales.length > 0) {
            state.currentScale = scales[0];
            scaleSelect.value = state.currentScale;
        }

    } catch (error) {
        console.error('Failed to load runs:', error);
        showError('Failed to load run data. Is the database available?');
    }
}

async function loadData() {
    try {
        const runTimestamp = state.currentRun === 'latest' ? null : state.currentRun;

        // Load all data in parallel
        const [overview, queries, aggregates] = await Promise.all([
            api.getOverview(state.currentScale, runTimestamp),
            api.getQueries(null, state.currentScale, runTimestamp),
            api.getAggregates(state.currentScale)
        ]);

        state.data.overview = overview;
        state.data.queries = queries;
        state.data.aggregates = aggregates;

        // Update last updated time
        document.getElementById('last-updated-time').textContent = new Date().toLocaleTimeString();

    } catch (error) {
        console.error('Failed to load data:', error);
        showError('Failed to load benchmark data');
    }
}

function setupEventListeners() {
    // Scale factor change
    document.getElementById('scale-select').addEventListener('change', async (e) => {
        state.currentScale = parseFloat(e.target.value);
        await loadData();
        renderDashboard();
    });

    // Run selector change
    document.getElementById('run-select').addEventListener('change', async (e) => {
        state.currentRun = e.target.value;
        await loadData();
        renderDashboard();
    });

    // Refresh button
    document.getElementById('refresh-btn').addEventListener('click', async () => {
        await loadRuns();
        await loadData();
        renderDashboard();
    });

    // Auto-refresh toggle
    document.getElementById('auto-refresh-toggle').addEventListener('change', (e) => {
        state.autoRefresh = e.target.checked;

        if (state.autoRefresh) {
            state.autoRefreshInterval = setInterval(async () => {
                await loadData();
                renderDashboard();
            }, 30000); // 30 seconds
        } else {
            if (state.autoRefreshInterval) {
                clearInterval(state.autoRefreshInterval);
                state.autoRefreshInterval = null;
            }
        }
    });

    // Sort by change
    document.getElementById('sort-by').addEventListener('change', () => {
        renderQueryComparison();
    });

    // Timeseries query selector
    document.getElementById('timeseries-query').addEventListener('change', () => {
        renderTimeseries();
    });

    // Timeseries metric selector
    document.getElementById('timeseries-metric').addEventListener('change', () => {
        renderTimeseries();
    });

    // Filter checkboxes for data table
    document.getElementById('filter-pgx').addEventListener('change', () => {
        renderDataTable();
    });

    document.getElementById('filter-native').addEventListener('change', () => {
        renderDataTable();
    });

    // Table search
    document.getElementById('table-search').addEventListener('input', (e) => {
        if (state.dataTable) {
            state.dataTable.setFilter([
                {field: 'query_name', type: 'like', value: e.target.value}
            ]);
        }
    });

    // Export CSV
    document.getElementById('export-csv-btn').addEventListener('click', () => {
        if (state.dataTable) {
            state.dataTable.download('csv', 'benchmark_data.csv');
        }
    });
}

function renderDashboard() {
    renderOverviewCards();
    renderQueryComparison();
    renderScalingAnalysis();
    renderHeatmap();
    renderScatterPlot();
    renderProfilingStats();
    populateTimeseriesQuerySelector();
    renderTimeseries();
    renderDataTable();
}

// ============================================================================
// Overview Cards
// ============================================================================

function renderOverviewCards() {
    const overview = state.data.overview;
    if (!overview) return;

    // PGX average
    document.getElementById('pgx-avg').textContent = overview.pgx_avg !== null
        ? overview.pgx_avg.toFixed(2)
        : '--';

    // Native average
    document.getElementById('native-avg').textContent = overview.native_avg !== null
        ? overview.native_avg.toFixed(2)
        : '--';

    // Speedup
    document.getElementById('speedup-value').textContent = overview.speedup !== null
        ? overview.speedup.toFixed(2)
        : '--';
}

// ============================================================================
// Query Performance Comparison (Horizontal Grouped Bar Chart)
// ============================================================================

function renderQueryComparison() {
    const queries = state.data.queries;
    if (!queries || queries.length === 0) return;

    // Group by query name
    const grouped = queries.reduce((acc, q) => {
        if (!acc[q.query_name]) {
            acc[q.query_name] = { pgx: null, native: null };
        }
        if (q.pgx_enabled) {
            acc[q.query_name].pgx = q.exec_time;
        } else {
            acc[q.query_name].native = q.exec_time;
        }
        return acc;
    }, {});

    // Convert to array and calculate speedup
    let data = Object.entries(grouped).map(([name, times]) => ({
        name,
        pgx: times.pgx || 0,
        native: times.native || 0,
        speedup: (times.native && times.pgx) ? times.native / times.pgx : 0
    }));

    // Sort based on user selection
    const sortBy = document.getElementById('sort-by').value;
    if (sortBy === 'speedup') {
        data.sort((a, b) => b.speedup - a.speedup);
    } else if (sortBy === 'absolute') {
        data.sort((a, b) => b.native - a.native);
    } else {
        data.sort((a, b) => a.name.localeCompare(b.name));
    }

    const labels = data.map(d => d.name);
    const pgxData = data.map(d => d.pgx);
    const nativeData = data.map(d => d.native);

    const ctx = document.getElementById('query-comparison-chart');
    if (state.charts.queryComparison) {
        state.charts.queryComparison.destroy();
    }

    state.charts.queryComparison = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [
                {
                    label: 'PGX Enabled',
                    data: pgxData,
                    backgroundColor: '#2ecc71',
                    borderColor: '#27ae60',
                    borderWidth: 1
                },
                {
                    label: 'Native PostgreSQL',
                    data: nativeData,
                    backgroundColor: '#3498db',
                    borderColor: '#2980b9',
                    borderWidth: 1
                }
            ]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: false
                },
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    callbacks: {
                        label: (context) => {
                            const label = context.dataset.label || '';
                            const value = context.parsed.x.toFixed(2);
                            return `${label}: ${value} ms`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Execution Time (ms)'
                    }
                }
            }
        }
    });
}

// ============================================================================
// Scaling Analysis (Multi-Line Chart)
// ============================================================================

function renderScalingAnalysis() {
    // For now, use placeholder data since we need cross-scale data
    // This will be populated when we have multiple scale factors
    const ctx = document.getElementById('scaling-chart');
    if (state.charts.scaling) {
        state.charts.scaling.destroy();
    }

    state.charts.scaling = new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['0.01', '0.1', '1.0'],
            datasets: [
                {
                    label: 'Q1 PGX',
                    data: [null, null, null],
                    borderColor: '#2ecc71',
                    backgroundColor: 'rgba(46, 204, 113, 0.1)',
                    tension: 0.4
                },
                {
                    label: 'Q1 Native',
                    data: [null, null, null],
                    borderColor: '#3498db',
                    backgroundColor: 'rgba(52, 152, 219, 0.1)',
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: false
                },
                legend: {
                    display: true
                }
            },
            scales: {
                y: {
                    title: {
                        display: true,
                        text: 'Execution Time (ms)'
                    },
                    beginAtZero: true
                },
                x: {
                    title: {
                        display: true,
                        text: 'Scale Factor'
                    }
                }
            }
        }
    });
}

// ============================================================================
// Performance Heatmap
// ============================================================================

async function renderHeatmap() {
    try {
        const data = await api.getHeatmap(state.currentScale, state.currentRun);

        const container = document.getElementById('heatmap-container');
        if (!data || !data.queries || data.queries.length === 0) {
            container.innerHTML = '<p class="placeholder">No data available for heatmap</p>';
            return;
        }

        // Create HTML table for heatmap
        const metricLabels = {
            exec_time: 'Exec Time',
            plan_time: 'Plan Time',
            memory: 'Memory',
            cache_hit: 'Cache Hit',
            disk_read: 'Disk Read',
            spilled: 'Spilled'
        };

        let html = '<table class="heatmap-table"><thead><tr><th>Query</th>';
        data.metrics.forEach(metric => {
            html += `<th>${metricLabels[metric] || metric}</th>`;
        });
        html += '</tr></thead><tbody>';

        // For each query, create row with both PGX and native
        data.queries.forEach(query => {
            ['pgx', 'native'].forEach(mode => {
                const rowKey = `${query}_${mode}`;
                const rowData = data.data[rowKey];

                if (!rowData) return;

                html += `<tr><td class="query-cell">${query} <span class="mode-badge ${mode}">${mode}</span></td>`;

                data.metrics.forEach(metric => {
                    const cell = rowData[metric];
                    if (cell) {
                        const colorClass = cell.color;
                        const value = cell.value;
                        const tooltip = `${metricLabels[metric]}: ${value} (${cell.percentile}th percentile)`;

                        html += `<td class="heatmap-cell ${colorClass}" title="${tooltip}" data-query="${query}" data-metric="${metric}" data-mode="${mode}">${value}</td>`;
                    } else {
                        html += '<td class="heatmap-cell empty">-</td>';
                    }
                });

                html += '</tr>';
            });
        });

        html += '</tbody></table>';
        container.innerHTML = html;

        // Add click handlers for cells
        container.querySelectorAll('.heatmap-cell:not(.empty)').forEach(cell => {
            cell.addEventListener('click', () => {
                const query = cell.dataset.query;
                const metric = cell.dataset.metric;
                const mode = cell.dataset.mode;

                // Map display names to API metric names
                const metricMap = {
                    exec_time: 'exec_time',
                    plan_time: 'planning_time',
                    memory: 'exec_time',  // Fallback to exec_time
                    cache_hit: 'exec_time',
                    disk_read: 'exec_time',
                    spilled: 'exec_time'
                };

                console.log(`Clicked: ${query} ${mode} - ${metric}`);
                // TODO: Show time series in modal or expand section
            });
        });

    } catch (error) {
        console.error('Error rendering heatmap:', error);
        document.getElementById('heatmap-container').innerHTML =
            '<p class="placeholder">Error loading heatmap</p>';
    }
}

// ============================================================================
// Profiling Overhead Summary
// ============================================================================

async function renderProfilingStats() {
    try {
        const data = await api.getProfileStats();
        const container = document.getElementById('profiling-stats');

        if (!data.stats || data.stats.length === 0) {
            container.innerHTML = '<p class="placeholder">No profiling data available. Run benchmarks with --profile flag.</p>';
            return;
        }

        // Group by pgx_enabled
        const pgxStats = data.stats.find(s => s.pgx_enabled);
        const nativeStats = data.stats.find(s => !s.pgx_enabled);

        let html = '<div class="profiling-summary">';

        // Helper to render stats section
        const renderSection = (stats, title) => {
            if (!stats) return '';

            const cpuCount = stats.cpu.count || 0;
            const heapCount = stats.heap.count || 0;
            const cpuRaw = stats.cpu.raw_kb || 0;
            const cpuComp = stats.cpu.compressed_kb || 0;
            const cpuRatio = stats.cpu.compression_ratio || 0;
            const totalKb = stats.total_kb || 0;
            const perQueryKb = stats.per_query_kb || 0;

            return `
                <div class="stat-card">
                    <h3>${title}</h3>
                    <div class="stat-row">
                        <span class="stat-label">CPU Profiles:</span>
                        <span class="stat-value">${cpuCount} queries</span>
                    </div>
                    ${cpuCount > 0 ? `
                    <div class="stat-row">
                        <span class="stat-label">Raw Size:</span>
                        <span class="stat-value">${cpuRaw.toFixed(1)} KB</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Compressed:</span>
                        <span class="stat-value">${cpuComp.toFixed(1)} KB (${cpuRatio.toFixed(1)}× compression)</span>
                    </div>
                    ` : ''}
                    ${heapCount > 0 ? `
                    <div class="stat-row">
                        <span class="stat-label">Heap Profiles:</span>
                        <span class="stat-value">${heapCount} queries</span>
                    </div>
                    ` : ''}
                    <div class="stat-row">
                        <span class="stat-label">Total Storage:</span>
                        <span class="stat-value">${totalKb.toFixed(1)} KB (${perQueryKb.toFixed(1)} KB/query)</span>
                    </div>
                </div>
            `;
        };

        html += renderSection(nativeStats, 'PostgreSQL (Native)');
        html += renderSection(pgxStats, 'pgx-lower (MLIR)');
        html += '</div>';

        container.innerHTML = html;

    } catch (error) {
        console.error('Error rendering profiling stats:', error);
        document.getElementById('profiling-stats').innerHTML =
            '<p class="placeholder">Error loading profiling stats</p>';
    }
}

// ============================================================================
// I/O & Memory Scatter Plot
// ============================================================================

function renderScatterPlot() {
    const queries = state.data.queries;
    if (!queries || queries.length === 0) return;

    // Prepare scatter data
    const pgxPoints = [];
    const nativePoints = [];

    queries.forEach(q => {
        const buffers = q.buffers || {};
        const totalBlocks = (buffers.shared_hit || 0) + (buffers.shared_read || 0);
        const cacheHitRatio = totalBlocks > 0
            ? ((buffers.shared_hit || 0) / totalBlocks) * 100
            : 0;

        // Estimate memory (buffers * 8KB block size / 1024 = MB)
        const memoryMB = totalBlocks * 8 / 1024;

        const point = {
            x: cacheHitRatio,
            y: memoryMB,
            r: Math.max(5, Math.min(20, q.exec_time / 50)), // Bubble size based on time
            label: q.query_name
        };

        if (q.pgx_enabled) {
            pgxPoints.push(point);
        } else {
            nativePoints.push(point);
        }
    });

    const ctx = document.getElementById('scatter-chart');
    if (state.charts.scatter) {
        state.charts.scatter.destroy();
    }

    state.charts.scatter = new Chart(ctx, {
        type: 'bubble',
        data: {
            datasets: [
                {
                    label: 'PGX Enabled',
                    data: pgxPoints,
                    backgroundColor: 'rgba(46, 204, 113, 0.6)',
                    borderColor: '#2ecc71',
                    borderWidth: 1
                },
                {
                    label: 'Native PostgreSQL',
                    data: nativePoints,
                    backgroundColor: 'rgba(52, 152, 219, 0.6)',
                    borderColor: '#3498db',
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true
                },
                tooltip: {
                    callbacks: {
                        label: (context) => {
                            const point = context.raw;
                            return [
                                `Query: ${point.label}`,
                                `Cache Hit: ${point.x.toFixed(1)}%`,
                                `Memory: ${point.y.toFixed(2)} MB`
                            ];
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Cache Hit Ratio (%)'
                    },
                    min: 0,
                    max: 100
                },
                y: {
                    title: {
                        display: true,
                        text: 'Memory Usage (MB)'
                    },
                    beginAtZero: true
                }
            }
        }
    });
}

// ============================================================================
// Time Series Evolution
// ============================================================================

function populateTimeseriesQuerySelector() {
    const queries = state.data.queries;
    if (!queries) return;

    const queryNames = [...new Set(queries.map(q => q.query_name))].sort();
    const select = document.getElementById('timeseries-query');

    select.innerHTML = '<option value="">Select query...</option>';
    queryNames.forEach(name => {
        const option = document.createElement('option');
        option.value = name;
        option.textContent = name;
        select.appendChild(option);
    });
}

async function renderTimeseries() {
    const query = document.getElementById('timeseries-query').value;
    const metric = document.getElementById('timeseries-metric').value;

    if (!query) {
        // Clear chart if no query selected
        if (state.charts.timeseries) {
            state.charts.timeseries.destroy();
            state.charts.timeseries = null;
        }
        return;
    }

    try {
        const data = await api.getTimeseries(query, metric);

        const labels = data.map(d => new Date(d.run_timestamp).toLocaleString());
        const pgxData = data.map(d => d.pgx);
        const nativeData = data.map(d => d.native);

        const ctx = document.getElementById('timeseries-chart');
        if (state.charts.timeseries) {
            state.charts.timeseries.destroy();
        }

        state.charts.timeseries = new Chart(ctx, {
            type: 'line',
            data: {
                labels,
                datasets: [
                    {
                        label: 'PGX Enabled',
                        data: pgxData,
                        borderColor: '#2ecc71',
                        backgroundColor: 'rgba(46, 204, 113, 0.1)',
                        tension: 0.4,
                        fill: true
                    },
                    {
                        label: 'Native PostgreSQL',
                        data: nativeData,
                        borderColor: '#3498db',
                        backgroundColor: 'rgba(52, 152, 219, 0.1)',
                        tension: 0.4,
                        fill: true
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true
                    },
                    title: {
                        display: true,
                        text: `${query} - ${metric}`
                    }
                },
                scales: {
                    y: {
                        title: {
                            display: true,
                            text: metric === 'exec_time' ? 'Execution Time (ms)' : 'Planning Time (ms)'
                        },
                        beginAtZero: true
                    }
                }
            }
        });

    } catch (error) {
        console.error('Failed to load timeseries:', error);
    }
}

// ============================================================================
// Raw Data Table (Tabulator.js)
// ============================================================================

function renderDataTable() {
    const queries = state.data.queries;
    if (!queries) return;

    // Filter data based on checkboxes
    const showPgx = document.getElementById('filter-pgx').checked;
    const showNative = document.getElementById('filter-native').checked;

    const filteredData = queries.filter(q => {
        if (q.pgx_enabled && !showPgx) return false;
        if (!q.pgx_enabled && !showNative) return false;
        return true;
    });

    // Destroy existing table if it exists
    if (state.dataTable) {
        state.dataTable.destroy();
    }

    // Create new table
    state.dataTable = new Tabulator('#data-table', {
        data: filteredData,
        layout: 'fitColumns',
        pagination: true,
        paginationSize: 20,
        paginationSizeSelector: [10, 20, 50, 100],
        columns: [
            {
                title: 'Query',
                field: 'query_name',
                sorter: 'string',
                width: 100
            },
            {
                title: 'PGX',
                field: 'pgx_enabled',
                sorter: 'boolean',
                width: 80,
                formatter: (cell) => cell.getValue() ? '✓' : '✗',
                hozAlign: 'center'
            },
            {
                title: 'Exec Time (ms)',
                field: 'exec_time',
                sorter: 'number',
                width: 120,
                formatter: (cell) => cell.getValue()?.toFixed(2) || '--'
            },
            {
                title: 'Plan Time (ms)',
                field: 'planning_time',
                sorter: 'number',
                width: 120,
                formatter: (cell) => cell.getValue()?.toFixed(2) || '--'
            },
            {
                title: 'Rows',
                field: 'row_count',
                sorter: 'number',
                width: 100
            },
            {
                title: 'Status',
                field: 'status',
                sorter: 'string',
                width: 100
            },
            {
                title: 'Timestamp',
                field: 'run_timestamp',
                sorter: 'datetime',
                width: 180,
                formatter: (cell) => {
                    const val = cell.getValue();
                    return val ? new Date(val).toLocaleString() : '--';
                }
            }
        ],
        initialSort: [
            {column: 'query_name', dir: 'asc'},
            {column: 'pgx_enabled', dir: 'desc'}
        ]
    });
}

// ============================================================================
// Error Handling
// ============================================================================

function showError(message) {
    console.error(message);
    // Could add a toast notification system here
    alert(message);
}
