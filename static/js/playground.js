const SAMPLE = {
  soil: {
    soil_texture: "loamy",
    soil_ph: 6.5,
    nitrogen: 80,
    phosphorus: 60,
    potassium: 45,
    soil_moisture_vwc: 50,
  },
  coordinates: { lat: -1.9441, lon: 30.0619 },
};

const requestBody = document.getElementById("requestBody");
const responseBody = document.getElementById("responseBody");

function loadSample() {
  requestBody.value = JSON.stringify(SAMPLE, null, 2);
}

document.getElementById("loadSample").addEventListener("click", loadSample);

document.getElementById("sendBtn").addEventListener("click", async () => {
  responseBody.textContent = "Sending…";
  let payload;
  try {
    payload = JSON.parse(requestBody.value);
  } catch (err) {
    responseBody.textContent = JSON.stringify(
      { error: "Invalid JSON: " + err.message },
      null,
      2
    );
    return;
  }

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    responseBody.textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    responseBody.textContent = JSON.stringify(
      { error: String(err.message || err) },
      null,
      2
    );
  }
});

loadSample();
