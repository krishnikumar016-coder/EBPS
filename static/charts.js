/**
 * Chart.js wrapper functions for the BurnGuard dashboard.
 * Creates styled charts with the dark theme color palette.
 */

// Global Chart.js defaults
Chart.defaults.color = 'rgba(196, 181, 253, 0.7)';
Chart.defaults.borderColor = 'rgba(46, 39, 68, 0.4)';
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.plugins.legend.labels.usePointStyle = true;
Chart.defaults.plugins.legend.labels.pointStyleWidth = 10;
Chart.defaults.plugins.legend.labels.padding = 16;
Chart.defaults.animation.duration = 1200;
Chart.defaults.animation.easing = 'easeOutQuart';

// Store chart instances for cleanup
const chartInstances = {};

function destroyChart(id) {
    if (chartInstances[id]) {
        chartInstances[id].destroy();
        delete chartInstances[id];
    }
}

/**
 * Risk Distribution Donut Chart
 */
function createRiskDistributionChart(canvasId, highRisk, mediumRisk, lowRisk) {
    destroyChart(canvasId);
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    // Default parameters fallback if mediumRisk omitted
    if (typeof mediumRisk === 'undefined') {
        mediumRisk = 0;
    }

    const total = highRisk + mediumRisk + lowRisk;
    const hasData = total > 0;

    chartInstances[canvasId] = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: hasData ? ['High Risk', 'Medium Risk', 'Low Risk'] : ['No Data'],
            datasets: [{
                data: hasData ? [highRisk, mediumRisk, lowRisk] : [1],
                backgroundColor: hasData
                    ? ['rgba(244, 63, 94, 0.85)', 'rgba(251, 191, 36, 0.85)', 'rgba(52, 211, 153, 0.85)']
                    : ['rgba(167, 139, 250, 0.08)'],
                borderColor: hasData
                    ? ['rgba(244, 63, 94, 0.4)', 'rgba(251, 191, 36, 0.4)', 'rgba(52, 211, 153, 0.4)']
                    : ['rgba(46, 39, 68, 0.3)'],
                borderWidth: 2,
                hoverOffset: 8,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '68%',
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: 'rgba(196, 181, 253, 0.8)',
                        font: { size: 12, weight: '500' },
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(19, 17, 28, 0.95)',
                    titleColor: '#f5f3ff',
                    bodyColor: 'rgba(196, 181, 253, 0.8)',
                    borderColor: 'rgba(46, 39, 68, 0.8)',
                    borderWidth: 1,
                    padding: 12,
                    cornerRadius: 8,
                    callbacks: {
                        label: function(ctx) {
                            if (!hasData) return 'No predictions yet';
                            const pct = ((ctx.raw / total) * 100).toFixed(1);
                            return `${ctx.label}: ${ctx.raw} (${pct}%)`;
                        }
                    }
                }
            },
        },
        plugins: [{
            id: 'centerText',
            afterDraw(chart) {
                const { width, height, ctx } = chart;
                ctx.save();
                const text = hasData ? total.toString() : '0';
                const subtext = 'Total';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                const centerY = height / 2 - 10;

                ctx.font = "800 28px 'Inter', sans-serif";
                ctx.fillStyle = '#f5f3ff';
                ctx.fillText(text, width / 2, centerY);

                ctx.font = "500 11px 'Inter', sans-serif";
                ctx.fillStyle = 'rgba(196, 181, 253, 0.5)';
                ctx.fillText(subtext, width / 2, centerY + 22);
                ctx.restore();
            }
        }]
    });
}

/**
 * Prediction Trend Line Chart
 */
function createPredictionTrendChart(canvasId, predictions) {
    destroyChart(canvasId);
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    // Take last 10 predictions, reversed for chronological order
    const recent = predictions.slice(0, 10).reverse();
    const labels = recent.map((p, i) => {
        if (p.employee_name) {
            return p.employee_name.split(' ')[0]; // First name only
        }
        return `#${i + 1}`;
    });
    const probs = recent.map(p => (p.burn_probability * 100).toFixed(1));

    // Gradient fill - Lavender Gradient
    const gradient = ctx.getContext('2d').createLinearGradient(0, 0, 0, 280);
    gradient.addColorStop(0, 'rgba(167, 139, 250, 0.35)');
    gradient.addColorStop(1, 'rgba(167, 139, 250, 0.0)');

    chartInstances[canvasId] = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Burn Probability (%)',
                data: probs,
                borderColor: '#a78bfa',
                backgroundColor: gradient,
                borderWidth: 2.5,
                fill: true,
                tension: 0.4,
                pointRadius: 4,
                pointHoverRadius: 7,
                pointBackgroundColor: '#c084fc',
                pointBorderColor: '#09090d',
                pointBorderWidth: 2,
            }, {
                label: 'Threshold (50%)',
                data: Array(labels.length).fill(50),
                borderColor: 'rgba(244, 63, 94, 0.4)',
                borderWidth: 1.5,
                borderDash: [6, 4],
                pointRadius: 0,
                fill: false,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { intersect: false, mode: 'index' },
            scales: {
                y: {
                    min: 0,
                    max: 100,
                    grid: { color: 'rgba(46, 39, 68, 0.3)' },
                    ticks: {
                        callback: v => v + '%',
                        font: { size: 11 },
                    },
                },
                x: {
                    grid: { display: false },
                    ticks: { font: { size: 11 } },
                }
            },
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        font: { size: 11, weight: '500' },
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(19, 17, 28, 0.95)',
                    titleColor: '#f5f3ff',
                    bodyColor: 'rgba(196, 181, 253, 0.8)',
                    borderColor: 'rgba(46, 39, 68, 0.8)',
                    borderWidth: 1,
                    padding: 12,
                    cornerRadius: 8,
                }
            }
        }
    });
}
