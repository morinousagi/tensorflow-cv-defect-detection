let model;

// 1. Load the model
async function init() {
    try {
        model = await tf.loadLayersModel('./model/model.json');
        document.getElementById('status').innerText = 'Model loaded. Upload an image!';
    } catch (err) {
        document.getElementById('status').innerText = 'Error loading model: ' + err;
    }
}

// 2. Handle Image Upload & Prediction
document.getElementById('imageUpload').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const imgElement = document.getElementById('imagePreview');
    imgElement.src = URL.createObjectURL(file);
    imgElement.style.display = 'block';

    imgElement.onload = async () => {
        const tensor = tf.browser.fromPixels(imgElement)
            .resizeNearestNeighbor([224, 224]) // Adjust to your model's input size
            .expandDims(0)
            .toFloat()
            .div(tf.scalar(255)); // Normalization

        const predictions = await model.predict(tensor).data();
        displayResults(predictions);
    };
});

function displayResults(predictions) {
    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = `Prediction: ${predictions}`;
}

init();
