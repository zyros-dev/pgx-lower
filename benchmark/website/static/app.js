const state = {
    currentScale: null,
    currentRun: null,
    data: {
        meta: null,
        raw: null,
        aggregate: null
    },
    charts: {}
};

const api = {
    async get(endpoint) {
        const response = await fetch(endpoint);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json();
    },

    getMeta() {
        return this.get('/api/meta');
    },

    getRaw(params = {}) {
        const query = new URLSearchParams(params).toString();
        return this.get(`/api/raw${query ? '?' + query : ''}`);
    },

    getAggregate(params = {}) {
        const query = new URLSearchParams(params).toString();
        return this.get(`/api/aggregate${query ? '?' + query : ''}`);
    },

    getProfiling(params = {}) {
        const query = new URLSearchParams(params).toString();
        return this.get(`/api/profiling${query ? '?' + query : ''}`);
    }
};

document.addEventListener('DOMContentLoaded', async () => {
    await loadData();
    setupEventListeners();
    updateRunSelectState();
    renderDashboard();
});

async function loadData() {
    state.data.meta = await api.getMeta();

    if (state.data.meta.latest) {
        state.currentRun = state.data.meta.latest.run_timestamp;
        state.currentScale = state.data.meta.latest.scale_factor;
    }

    populateScaleSelector();

    const params = {};
    if (state.currentRun) params.run_timestamp = state.currentRun;
    if (state.currentScale) params.scale_factor = state.currentScale;

    state.data.raw = await api.getRaw(params);
    state.data.aggregate = await api.getAggregate(params);

    updateTimestamp();
}

function populateScaleSelector() {
    const select = document.getElementById('scale-select');
    select.innerHTML = state.data.meta.scale_factors
        .map(sf => `<option value="${sf}" ${sf === state.currentScale ? 'selected' : ''}>${sf}</option>`)
        .join('');
}

function updateRunSelectState() {
    const viewSelect = document.getElementById('view-select');
    const runSelect = document.getElementById('run-select');
    const isAggregate = viewSelect.value === 'aggregate';

    runSelect.disabled = isAggregate;
    if (isAggregate) {
        runSelect.style.opacity = '0.5';
        runSelect.style.cursor = 'not-allowed';
    } else {
        runSelect.style.opacity = '1';
        runSelect.style.cursor = 'pointer';
    }
}

function setupEventListeners() {
    document.getElementById('view-select').addEventListener('change', () => {
        updateRunSelectState();
        renderDashboard();
    });

    document.getElementById('scale-select').addEventListener('change', async (e) => {
        state.currentScale = parseFloat(e.target.value);
        await loadData();
        renderDashboard();
    });

    document.getElementById('sort-by').addEventListener('change', () => {
        renderQueryComparison();
    });

    document.getElementById('show-pgx').addEventListener('change', () => {
        renderScalingAnalysis();
    });

    document.getElementById('show-native').addEventListener('change', () => {
        renderScalingAnalysis();
    });

    document.getElementById('flame-query').addEventListener('change', () => {
        renderFlameChart();
    });

    document.getElementById('flame-profile-type').addEventListener('change', () => {
        renderFlameChart();
    });

    document.getElementById('flame-inverted').addEventListener('change', () => {
        renderFlameChart();
    });

    document.getElementById('flame-min-width').addEventListener('change', () => {
        renderFlameChart();
    });

    document.querySelectorAll('input[name="flame-mode"]').forEach(radio => {
        radio.addEventListener('change', () => {
            renderFlameChart();
        });
    });
}

function updateTimestamp() {
    const now = new Date().toLocaleTimeString();
    document.getElementById('last-updated-time').textContent = now;
}

function renderDashboard() {
    renderOverviewCards();
    renderQueryComparison();
    renderScalingAnalysis();
    renderHeatmap();
    populateFlameQuerySelector();
    renderFlameChart();
}

function renderOverviewCards() {
    const agg = state.data.aggregate;
    if (!agg || agg.length === 0) return;

    const pgxData = agg.find(a => a.pgx_enabled);
    const nativeData = agg.find(a => !a.pgx_enabled);

    document.getElementById('pgx-avg').textContent =
        pgxData?.avg_exec_time?.toFixed(2) || '--';

    document.getElementById('native-avg').textContent =
        nativeData?.avg_exec_time?.toFixed(2) || '--';

    if (pgxData && nativeData && pgxData.avg_exec_time && nativeData.avg_exec_time) {
        const speedup = nativeData.avg_exec_time / pgxData.avg_exec_time;
        document.getElementById('speedup-value').textContent = speedup.toFixed(2);
    } else {
        document.getElementById('speedup-value').textContent = '--';
    }
}

function renderQueryComparison() {
    const raw = state.data.raw;
    if (!raw || raw.length === 0) return;

    const grouped = raw.reduce((acc, q) => {
        if (!acc[q.query_name]) acc[q.query_name] = {};
        acc[q.query_name][q.pgx_enabled ? 'pgx' : 'native'] = q.exec_time;
        return acc;
    }, {});

    let data = Object.entries(grouped).map(([name, times]) => ({
        name,
        pgx: times.pgx || 0,
        native: times.native || 0,
        speedup: times.native && times.pgx ? times.native / times.pgx : 0
    }));

    const sortBy = document.getElementById('sort-by').value;
    if (sortBy === 'speedup') data.sort((a, b) => b.speedup - a.speedup);
    else if (sortBy === 'absolute') data.sort((a, b) => b.native - a.native);
    else data.sort((a, b) => a.name.localeCompare(b.name));

    const container = document.getElementById('query-comparison-chart');
    if (!state.charts.queryComparison) {
        state.charts.queryComparison = echarts.init(container);
    }

    state.charts.queryComparison.setOption({
        backgroundColor: 'transparent',
        tooltip: {
            trigger: 'axis',
            axisPointer: {type: 'shadow'},
            backgroundColor: 'rgba(255, 255, 255, 0.95)',
            borderColor: '#e5e7eb',
            textStyle: {color: '#1f2937'}
        },
        legend: {
            data: ['PGX', 'Native'],
            textStyle: {color: '#6b7280'},
            top: 10
        },
        grid: {left: '8%', right: '5%', bottom: '10%', top: '15%', containLabel: true},
        xAxis: {
            type: 'category',
            data: data.map(d => d.name),
            axisLine: {lineStyle: {color: '#e5e7eb'}},
            axisLabel: {color: '#6b7280'},
            splitLine: {show: false}
        },
        yAxis: {
            type: 'value',
            name: 'Execution Time (ms)',
            nameTextStyle: {color: '#6b7280'},
            axisLine: {lineStyle: {color: '#e5e7eb'}},
            axisLabel: {color: '#6b7280'},
            splitLine: {lineStyle: {color: '#f3f4f6', type: 'solid'}}
        },
        series: [
            {
                name: 'PGX',
                type: 'bar',
                data: data.map(d => d.pgx),
                itemStyle: {color: '#f97316', borderRadius: [2, 2, 0, 0]}
            },
            {
                name: 'Native',
                type: 'bar',
                data: data.map(d => d.native),
                itemStyle: {color: '#3b82f6', borderRadius: [2, 2, 0, 0]}
            }
        ]
    });
}

async function renderScalingAnalysis() {
    const container = document.getElementById('scaling-chart');

    const allRaw = await api.getRaw();

    const byQuery = allRaw.reduce((acc, q) => {
        if (!acc[q.query_name]) acc[q.query_name] = {};
        if (!acc[q.query_name][q.scale_factor]) {
            acc[q.query_name][q.scale_factor] = {};
        }
        acc[q.query_name][q.scale_factor][q.pgx_enabled ? 'pgx' : 'native'] = q.exec_time;
        return acc;
    }, {});

    const firstQuery = Object.keys(byQuery)[0];
    if (!firstQuery) return;

    const scaleFactors = Object.keys(byQuery[firstQuery]).map(Number).sort((a, b) => a - b);
    const pgxData = scaleFactors.map(sf => byQuery[firstQuery][sf]?.pgx || null);
    const nativeData = scaleFactors.map(sf => byQuery[firstQuery][sf]?.native || null);

    const showPgx = document.getElementById('show-pgx').checked;
    const showNative = document.getElementById('show-native').checked;

    if (!state.charts.scaling) {
        state.charts.scaling = echarts.init(container);
    }

    const series = [];
    if (showPgx) {
        series.push({
            name: `${firstQuery} PGX`,
            type: 'line',
            data: pgxData,
            smooth: true,
            itemStyle: {color: '#f97316'},
            lineStyle: {width: 2},
            symbol: 'circle',
            symbolSize: 6
        });
    }
    if (showNative) {
        series.push({
            name: `${firstQuery} Native`,
            type: 'line',
            data: nativeData,
            smooth: true,
            itemStyle: {color: '#3b82f6'},
            lineStyle: {width: 2},
            symbol: 'circle',
            symbolSize: 6
        });
    }

    state.charts.scaling.setOption({
        backgroundColor: 'transparent',
        tooltip: {
            trigger: 'axis',
            backgroundColor: 'rgba(255, 255, 255, 0.95)',
            borderColor: '#e5e7eb',
            textStyle: {color: '#1f2937'}
        },
        legend: {
            data: series.map(s => s.name),
            textStyle: {color: '#6b7280'},
            top: 10
        },
        grid: {left: '8%', right: '5%', bottom: '10%', top: '15%', containLabel: true},
        xAxis: {
            type: 'category',
            data: scaleFactors,
            name: 'Scale Factor',
            nameLocation: 'middle',
            nameGap: 30,
            nameTextStyle: {color: '#6b7280'},
            axisLine: {lineStyle: {color: '#e5e7eb'}},
            axisLabel: {color: '#6b7280'},
            splitLine: {show: false}
        },
        yAxis: {
            type: 'value',
            name: 'Execution Time (ms)',
            nameLocation: 'middle',
            nameGap: 50,
            nameTextStyle: {color: '#6b7280'},
            axisLine: {lineStyle: {color: '#e5e7eb'}},
            axisLabel: {color: '#6b7280'},
            splitLine: {lineStyle: {color: '#f3f4f6', type: 'solid'}}
        },
        series
    });
}

async function renderHeatmap() {
    const raw = state.data.raw;
    if (!raw || raw.length === 0) return;

    const container = document.getElementById('heatmap-container');
    const currentView = document.getElementById('view-select').value;

    const queries = [...new Set(raw.map(r => r.query_name))].sort();

    // Different metrics for aggregate vs raw view
    const metrics = currentView === 'raw'
        ? [
            { key: 'result_hash', label: 'Hash', format: v => v || '-', color: false },
            { key: 'exec_time', label: 'Exec (ms)', format: v => v.toFixed(2) },
            { key: 'planning_time', label: 'Plan (ms)', format: v => v.toFixed(2) },
            { key: 'cpu_total_sec', label: 'CPU (s)', format: v => v.toFixed(3) },
            { key: 'memory_peak_mb', label: 'Mem (MB)', format: v => v.toFixed(1) },
            { key: 'shared_hit', label: 'Buffers', format: v => v.toString() }
          ]
        : [
            { key: 'exec_time', label: 'Exec (ms)', format: v => v.toFixed(2) },
            { key: 'planning_time', label: 'Plan (ms)', format: v => v.toFixed(2) },
            { key: 'cpu_total_sec', label: 'CPU (s)', format: v => v.toFixed(3) },
            { key: 'cpu_percent', label: 'CPU %', format: v => v.toFixed(1) + '%' },
            { key: 'memory_peak_mb', label: 'Mem (MB)', format: v => v.toFixed(1) },
            { key: 'shared_hit', label: 'Buffers', format: v => v.toString() }
          ];

    // Collect data by query × mode
    const data = {};
    raw.forEach(r => {
        const key = `${r.query_name}_${r.pgx_enabled ? 'pgx' : 'native'}`;
        data[key] = r;
    });

    // Calculate min/max for each metric for color scaling
    const ranges = {};
    metrics.forEach(m => {
        const values = Object.values(data).map(d => d[m.key] || 0).filter(v => v > 0);
        ranges[m.key] = {
            min: Math.min(...values),
            max: Math.max(...values)
        };
    });

    // Helper to get color based on value (green=good, red=bad)
    function getColor(metric, value) {
        // Skip color for non-numeric metrics
        if (metric.color === false || !value || value === 0) return 'transparent';

        const range = ranges[metric.key];
        if (!range || range.min === range.max) return 'rgba(100, 100, 100, 0.3)';

        // Normalize value to 0-1
        const normalized = (value - range.min) / (range.max - range.min);

        // For all metrics: low is good (green), high is bad (red)
        const hue = (1 - normalized) * 120;
        return `hsla(${hue}, 70%, 50%, 0.4)`;
    }

    // Render table
    let html = '<table class="heatmap-table"><thead><tr><th>Query</th>';
    metrics.forEach(m => html += `<th>${m.label}</th>`);
    html += '</tr></thead><tbody>';

    queries.forEach(query => {
        ['native', 'pgx'].forEach(mode => {
            const key = `${query}_${mode}`;
            const rowData = data[key];
            if (!rowData) return;

            html += `<tr><td class="query-cell">${query} <span class="mode-badge ${mode}">${mode}</span></td>`;
            metrics.forEach(metric => {
                const value = rowData[metric.key];
                const color = getColor(metric, value);
                const displayValue = value !== undefined && value !== null ? metric.format(value) : '-';
                html += `<td class="heatmap-cell" style="background-color: ${color}">${displayValue}</td>`;
            });
            html += '</tr>';
        });
    });

    html += '</tbody></table>';
    container.innerHTML = html;
}

function mergeFlameNodes(target, source) {
    // Add source value to target
    target.value = (target.value || 0) + (source.value || 0);

    if (!source.children || source.children.length === 0) {
        return;
    }

    // Initialize children if needed
    if (!target.children) {
        target.children = [];
    }

    // Merge each source child
    for (const sourceChild of source.children) {
        // Find matching child in target
        let targetChild = target.children.find(c => c.name === sourceChild.name);

        if (!targetChild) {
            // Create new child (deep copy)
            targetChild = { name: sourceChild.name, value: 0, children: [] };
            target.children.push(targetChild);
        }

        // Recursively merge
        mergeFlameNodes(targetChild, sourceChild);
    }
}

function populateFlameQuerySelector() {
    const queries = [...new Set(state.data.raw.map(r => r.query_name))].sort();
    const select = document.getElementById('flame-query');

    select.innerHTML = '<option value="__AGGREGATE__">All Queries (Aggregated)</option>' +
        queries.map(q => `<option value="${q}">${q}</option>`).join('');

    // Default to aggregate view
    select.value = '__AGGREGATE__';
}

async function renderFlameChart() {
    const query = document.getElementById('flame-query').value;
    const mode = document.querySelector('input[name="flame-mode"]:checked').value;
    const profileType = document.getElementById('flame-profile-type').value;
    const container = document.getElementById('flame-chart-container');

    if (!query) {
        container.innerHTML = '<p class="placeholder">Select a query to view flame chart</p>';
        return;
    }

    const pgxEnabled = mode === 'pgx';
    const fieldName = profileType === 'cpu' ? 'cpu_flamegraph_lz4' : 'heap_flamegraph_lz4';
    const profileLabel = profileType === 'cpu' ? 'CPU' : 'Memory';

    try {
        let flameData;

        if (query === '__AGGREGATE__') {
            // Fetch all queries and merge their flamegraphs
            const rawData = await api.getRaw({
                pgx_enabled: pgxEnabled,
                run_timestamp: state.currentRun,
                include_flamegraph: true
            });

            if (!rawData || rawData.length === 0) {
                container.innerHTML = `<p class="placeholder">No ${profileLabel} profiles available</p>`;
                return;
            }

            // Merge all flamegraphs
            const merged = { name: 'root', value: 0, children: [] };
            for (const row of rawData) {
                if (!row[fieldName]) continue;

                // Decompress individual flamegraph
                const base64Data = row[fieldName];
                const binaryString = atob(base64Data);
                const bytes = new Uint8Array(binaryString.length);
                for (let i = 0; i < binaryString.length; i++) {
                    bytes[i] = binaryString.charCodeAt(i);
                }
                const decompressed = LZ4.decompress(bytes);
                const jsonString = new TextDecoder().decode(decompressed);
                const queryFlame = JSON.parse(jsonString);

                // Merge into aggregate
                mergeFlameNodes(merged, queryFlame);
            }

            flameData = merged;
        } else {
            // Single query flamegraph
            const rawData = await api.getRaw({
                query_name: query,
                pgx_enabled: pgxEnabled,
                run_timestamp: state.currentRun,
                include_flamegraph: true
            });

            if (!rawData || rawData.length === 0 || !rawData[0][fieldName]) {
                container.innerHTML = `<p class="placeholder">No ${profileLabel} profile available for this query</p>`;
                return;
            }

            // Decompress LZ4 data from base64
            const base64Data = rawData[0][fieldName];
            const binaryString = atob(base64Data);
            const bytes = new Uint8Array(binaryString.length);
            for (let i = 0; i < binaryString.length; i++) {
                bytes[i] = binaryString.charCodeAt(i);
            }

            // Decompress with LZ4
            const decompressed = LZ4.decompress(bytes);
            const jsonString = new TextDecoder().decode(decompressed);
            flameData = JSON.parse(jsonString);
        }

        const inverted = document.getElementById('flame-inverted').checked;
        const minFrameSize = parseInt(document.getElementById('flame-min-width').value);

        container.innerHTML = '<div id="flame-chart"></div>';
        const chart = flamegraph()
            .width(container.offsetWidth)
            .cellHeight(18)
            .transitionDuration(750)
            .minFrameSize(minFrameSize)
            .transitionEase(d3.easeCubic)
            .sort(true)
            .inverted(inverted)
            .selfValue(false);

        d3.select('#flame-chart')
            .datum(flameData)
            .call(chart);

    } catch (error) {
        container.innerHTML = `<p class="placeholder">Error loading flame chart: ${error.message}</p>`;
        console.error('Flamegraph error:', error);
    }
}
