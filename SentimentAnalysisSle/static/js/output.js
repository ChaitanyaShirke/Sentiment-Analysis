// static/scripts.js
document.addEventListener('DOMContentLoaded', function() {
    // Handle form submission
    document.querySelector('form').addEventListener('submit', function(event) {
        const fileInput = document.getElementById('review_file');
        const file = fileInput.files[0];
        if (file && file.type !== 'text/csv') {
            alert('Please upload a valid CSV file.');
            event.preventDefault();
        }
    });

    // Add event listener for category selection
    const categorySelect = document.getElementById('category');
    categorySelect.addEventListener('change', function() {
        const selectedCategory = this.value;
        if (selectedCategory) {
            // Implement filter logic based on selected category
            // For now, just log the selected category
            console.log('Selected category:', selectedCategory);
            // You may want to add AJAX requests here to dynamically update the charts
        }
    });

    // Function to show a loading spinner
    function showLoadingSpinner() {
        document.body.insertAdjacentHTML('beforeend', 
            <div id="loading-spinner" class="d-flex justify-content-center align-items-center" style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(255, 255, 255, 0.8); z-index: 9999;">
                <div class="spinner-border text-primary" role="status">
                    <span class="sr-only">Loading...</span>
                </div>
            </div>
        );
    }

    // Function to hide the loading spinner
    function hideLoadingSpinner() {
        const spinner = document.getElementById('loading-spinner');
        if (spinner) {
            spinner.remove();
        }
    }

    // Show loading spinner when the form is submitted
    document.querySelector('form').addEventListener('submit', function() {
        showLoadingSpinner();
    });

    // Hide loading spinner after the page is loaded
    window.addEventListener('load', function() {
        hideLoadingSpinner();
    });

    // Initialize tooltips for charts
    function initializeTooltips() {
        // Assuming you use a chart library that supports tooltips
        const charts = document.querySelectorAll('.chart');
        charts.forEach(chart => {
            chart.addEventListener('mouseover', function(event) {
                // Display tooltip with detailed information
                // Example logic, adjust based on your chart library
                console.log('Chart hovered:', event.target);
            });
        });
    }

    initializeTooltips();
});