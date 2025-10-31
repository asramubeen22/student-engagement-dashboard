// Sidebar toggle functionality
document.addEventListener('DOMContentLoaded', function() {
    const sidebarToggle = document.querySelector('.sidebar-toggle');
    const sidebar = document.querySelector('.sidebar');
    
    if (sidebarToggle) {
        sidebarToggle.addEventListener('click', function() {
            sidebar.classList.toggle('collapsed');
        });
    }
    
    // Load initial data for dashboard
    if (window.location.pathname === '/') {
        loadDashboardData();
    }
});

// Dashboard data loading
async function loadDashboardData() {
    try {
        const response = await fetch('/api/dashboard-stats');
        const data = await response.json();
        
        // Update stats cards
        document.getElementById('total-students').textContent = data.total_students.toLocaleString();
        document.getElementById('at-risk').textContent = data.at_risk.toLocaleString();
        document.getElementById('intervention-rate').textContent = data.intervention_rate + '%';
        document.getElementById('avg-attendance').textContent = data.avg_attendance + '%';
        
    } catch (error) {
        console.error('Error loading dashboard data:', error);
    }
}

// Risk gauge creation
function createRiskGauge(score, elementId = 'risk-gauge') {
    const data = [
        {
            type: "indicator",
            mode: "gauge+number",
            value: score,
            title: { text: "Engagement Risk", font: { size: 16 } },
            gauge: {
                axis: { range: [0, 100] },
                bar: { color: "#4361ee" },
                steps: [
                    { range: [0, 30], color: "#10b981" },
                    { range: [30, 70], color: "#f59e0b" },
                    { range: [70, 100], color: "#ef4444" }
                ],
                threshold: {
                    line: { color: "red", width: 4 },
                    thickness: 0.75,
                    value: 70
                }
            }
        }
    ];

    const layout = {
        width: 300,
        height: 250,
        margin: { t: 50, r: 0, b: 0, l: 0 },
        font: { family: "Inter, sans-serif" }
    };

    Plotly.newPlot(elementId, data, layout, { displayModeBar: false });
}