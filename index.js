document.getElementById('imageInput').addEventListener('change', function(e) {
    const file = e.target.files[0];
    const fileName = document.getElementById('fileName');
    const preview = document.getElementById('imagePreview');
    const uploadIcon = document.querySelector('.upload-area svg');
    const uploadText = document.querySelector('.upload-area p');

    if (file) {
        fileName.textContent = file.name;
        const reader = new FileReader();
        reader.onload = function(e) {
            preview.src = e.target.result;
            preview.style.display = 'block';
            uploadIcon.style.display = 'none';
            uploadText.style.display = 'none';
        }
        reader.readAsDataURL(file);
    }
});

document.getElementById('predictForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const loadingOverlay = document.getElementById('loadingOverlay');
    const resultsSection = document.getElementById('resultsSection');
    loadingOverlay.style.display = 'flex';
    resultsSection.style.display = 'none';

    const formData = new FormData(e.target);

    try {
        const response = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Server error');
        }

        const result = await response.json();

        // Update texture
        const textureBadge = document.getElementById('textureBadge');
        textureBadge.textContent = (result.soil_texture.charAt(0).toUpperCase() + result.soil_texture.slice(1)) + " Soil";

        // Update crop
        const cropResult = document.getElementById('cropResult');
        const bestCrop = result.crop_recommendations[0];
        cropResult.textContent = `${bestCrop.crop} (Suitability: ${bestCrop.suitability_score}%)`;

        // Update fertilizer
        const fertilizerResult = document.getElementById('fertilizerResult');
        const fert = result.fertilizer_recommendation;
        fertilizerResult.innerHTML = `<strong>${fert.recommended_fertilizer}</strong><br><small>${fert.description}</small>`;

        // Update tips
        const tipsList = document.getElementById('tipsList');
        tipsList.innerHTML = '';
        if (fert.additional_recommendations && fert.additional_recommendations.length > 0) {
            fert.additional_recommendations.forEach(tip => {
                const li = document.createElement('li');
                li.style.marginTop = '0.5rem';
                li.textContent = "• " + tip;
                tipsList.appendChild(li);
            });
        } else {
            tipsList.innerHTML = '<li>• No additional adjustments needed.</li>';
        }

        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth' });

    } catch (error) {
        alert("Error: " + error.message);
    } finally {
        loadingOverlay.style.display = 'none';
    }
});
