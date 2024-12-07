document.addEventListener('DOMContentLoaded', function() {
    const form = document.querySelector('form');
    const resultList = document.getElementById('result-list');
    const textQueryInput = document.getElementById('text-query');
    const imageQueryInput = document.getElementById('image-query');
    const lambdaInput = document.getElementById('lambda');
    const lambdaDisplay = document.getElementById('lambda-display');
    const embeddingChoiceInput = document.getElementById('embedding-choice');
    const pcaKGroup = document.getElementById('pca-k-group');

    // Update lambda display when slider moves
    lambdaInput.addEventListener('input', function() {
        lambdaDisplay.textContent = `Text Query Weight: ${this.value}`;
    });

    // Trigger initial display
    lambdaDisplay.textContent = `Text Query Weight: ${lambdaInput.value}`;

    // Toggle PCA k-group visibility
    embeddingChoiceInput.addEventListener('change', function() {
        if (this.value === 'pca') {
            pcaKGroup.style.display = 'block';
        } else {
            pcaKGroup.style.display = 'none';
        }
    });

    // Initial check
    if (embeddingChoiceInput.value === 'pca') {
        pcaKGroup.style.display = 'block';
    } else {
        pcaKGroup.style.display = 'none';
    }

    form.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Clear previous results
        resultList.innerHTML = 'Searching...';

        // Create FormData object
        const formData = new FormData();

        // Add text query if provided
        if (textQueryInput.value) {
            formData.append('text_query', textQueryInput.value);
        }

        // Add image file if provided
        if (imageQueryInput.files.length > 0) {
            formData.append('image_query', imageQueryInput.files[0]);
        }

        // Add lambda value
        formData.append('lambda', lambdaInput.value);

        // Add embedding choice
        formData.append('embedding_choice', embeddingChoiceInput.value);

        // Add PCA k value (if applicable)
        formData.append('pca_k', document.getElementById('pca-k').value);

        // Send AJAX request
        fetch('/search', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // Clear previous results
            resultList.innerHTML = '';

            if (data.results && data.results.length > 0) {
                // Create result grid
                const resultGrid = document.createElement('div');
                resultGrid.className = 'result-grid';

                // Populate results
                data.results.forEach((result, index) => {
                    const resultItem = document.createElement('div');
                    resultItem.className = 'result-item';

                    const img = document.createElement('img');
                    img.src = result.image_path;
                    img.alt = `Result ${index + 1}`;

                    const similarityText = document.createElement('p');
                    similarityText.textContent = `Similarity: ${(result.similarity * 100).toFixed(2)}%`;

                    resultItem.appendChild(img);
                    resultItem.appendChild(similarityText);
                    resultGrid.appendChild(resultItem);
                });

                resultList.appendChild(resultGrid);
            } else {
                resultList.textContent = 'No results found.';
            }
        })
        .catch(error => {
            console.error('Error:', error);
            resultList.textContent = 'An error occurred during search.';
        });
    });
});
