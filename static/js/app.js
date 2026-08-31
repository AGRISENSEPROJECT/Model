const API_BASE = window.AGRISENSE_API_BASE || "";

document.getElementById("imageInput").addEventListener("change", function (e) {
  const file = e.target.files[0];
  const fileName = document.getElementById("fileName");
  const preview = document.getElementById("imagePreview");
  const uploadIcon = document.querySelector(".upload-area svg");
  const uploadText = document.querySelector(".upload-area p");

  if (file) {
    fileName.textContent = file.name;
    const reader = new FileReader();
    reader.onload = function (ev) {
      preview.src = ev.target.result;
      preview.style.display = "block";
      if (uploadIcon) uploadIcon.style.display = "none";
      if (uploadText) uploadText.style.display = "none";
    };
    
    reader.readAsDataURL(file);
  }
});

document.getElementById("predictForm").addEventListener("submit", async (e) => {
  e.preventDefault();

  const loadingOverlay = document.getElementById("loadingOverlay");
  const resultsSection = document.getElementById("resultsSection");
  loadingOverlay.style.display = "flex";
  resultsSection.style.display = "none";

  const formData = new FormData(e.target);

  try {
    const response = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.error || "Server error");
    }

    const result = await response.json();
    const texture =
      result.soil_texture ||
      (result.soil_analysis && result.soil_analysis.texture) ||
      "unknown";
    const textureConf =
      result.soil_analysis && result.soil_analysis.confidence != null
        ? ` (${Math.round(result.soil_analysis.confidence * 100)}%)`
        : "";

    document.getElementById("textureBadge").textContent =
      texture.charAt(0).toUpperCase() + texture.slice(1) + " Soil" + textureConf;

    const crops = result.crop_recommendations || [];
    const bestCrop = crops[0];
    const cropResult = document.getElementById("cropResult");
    if (bestCrop) {
      const yieldBit =
        bestCrop.predicted_yield != null
          ? ` · pred. yield ${bestCrop.predicted_yield}`
          : "";
      cropResult.textContent = `${bestCrop.crop} — ${bestCrop.suitability_score}% (${bestCrop.source || "ml"})${yieldBit}`;
    } else {
      cropResult.textContent = "No recommendation";
    }

    const fert = result.fertilizer_recommendation || {};
    const fertilizerResult = document.getElementById("fertilizerResult");
    if (fert.recommended_fertilizer) {
      fertilizerResult.innerHTML = `<strong>${fert.recommended_fertilizer}</strong><br><small>${fert.description || ""}</small>`;
    } else {
      fertilizerResult.textContent = fert.message || fert.error || "Unavailable";
    }

    const tipsList = document.getElementById("tipsList");
    tipsList.innerHTML = "";
    const tips = fert.additional_recommendations || [];
    if (tips.length) {
      tips.forEach((tip) => {
        const li = document.createElement("li");
        li.style.marginTop = "0.5rem";
        li.textContent = "• " + tip;
        tipsList.appendChild(li);
      });
    } else {
      tipsList.innerHTML = "<li>• No additional adjustments needed.</li>";
    }

    resultsSection.style.display = "block";
    resultsSection.scrollIntoView({ behavior: "smooth" });
  } catch (error) {
    alert("Error: " + error.message);
  } finally {
    loadingOverlay.style.display = "none";
  }
});
