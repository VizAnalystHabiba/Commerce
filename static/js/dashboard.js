// static/js/dashboard.js — Commerce360 AI Dashboard Client Functions

document.addEventListener('DOMContentLoaded', function () {
    // Mobile sidebar toggle
    const sidebarToggle = document.getElementById('sidebarToggle');
    if (sidebarToggle) {
        sidebarToggle.addEventListener('click', function (e) {
            e.preventDefault();
            const wrapper = document.getElementById('wrapper');
            if (wrapper) {
                wrapper.classList.toggle('toggled');
            }
        });
    }

    // Auto-resize Plotly charts on window resize
    window.addEventListener('resize', function () {
        const plotElements = document.querySelectorAll('.js-plotly-plot');
        plotElements.forEach(function (plotEl) {
            if (window.Plotly && plotEl) {
                window.Plotly.Plots.resize(plotEl);
            }
        });
    });
});
