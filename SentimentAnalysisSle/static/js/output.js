document.addEventListener('DOMContentLoaded', function() {
    // Handle form submission
    const form = document.querySelector('form');
    const fileInput = document.getElementById('review_file');
    
    form.addEventListener('submit', function(event) {
        const file = fileInput.files[0];
        if (file && file.type !== 'text/csv') {
            alert('Please upload a valid CSV file.');
            event.preventDefault();
        } else {
            showLoadingSpinner(); // Show spinner when form is submitted
        }
    });

    // Add event listener for category selection
    const categorySelect = document.getElementById('category');
    
    categorySelect.addEventListener('change', function() {
        const selectedCategory = this.value;
        if (selectedCategory) {
            // Implement filter logic based on selected category
            fetchChartData(selectedCategory); // AJAX call to update charts
        }
    });

    // Function to show a loading spinner
    function showLoadingSpinner() {
        document.body.insertAdjacentHTML('beforeend', 
            `<div id="loading-spinner" class="d-flex justify-content-center align-items-center position-fixed top-0 start-0 w-100 h-100 bg-light bg-opacity-75 zindex-9999">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
            </div>`
        );
    }

    // Function to hide the loading spinner
    function hideLoadingSpinner() {
        const spinner = document.getElementById('loading-spinner');
        if (spinner) {
            spinner.remove();
        }
    }

    // Hide loading spinner after the page is loaded
    window.addEventListener('load', function() {
        hideLoadingSpinner();
    });

    // Function to fetch chart data based on selected category
    function fetchChartData(category) {
        fetch(`/get-chart-data?category=${encodeURIComponent(category)}`)
            .then(response => response.json())
            .then(data => {
                updateCharts(data); // Update charts with new data
            })
            .catch(error => {
                console.error('Error fetching chart data:', error);
            });
    }

    // Function to update charts with new data
    function updateCharts(data) {
        // Assuming you have chart elements with IDs corresponding to chart types
        // Example:
        const pieChart = document.getElementById('pie-chart');
        const barChart = document.getElementById('bar-chart');
        // Update chart sources
        pieChart.src = data.pieChartUrl;
        barChart.src = data.barChartUrl;
    }

    // Initialize tooltips for charts
    function initializeTooltips() {
        // Assuming you use a chart library that supports tooltips
        const charts = document.querySelectorAll('.chart');
        charts.forEach(chart => {
            chart.addEventListener('mouseover', function(event) {
                // Display tooltip with detailed information
                // Example logic, adjust based on your chart library
                console.log('Chart hovered:', event.target);
                // Display tooltip here
            });
        });
    }

    initializeTooltips();
});
