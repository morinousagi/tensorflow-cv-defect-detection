let model;
const modelURL = './model.json'; // Ensure this matches your file name

async function loadModel() {
    try {
        console.log("Loading model...");
        // Ensure tf is available before this call
        model = await tf.loadLayersModel(modelURL);
        console.log("Model loaded successfully!");
    } catch (error) {
        console.error("Error loading model:", error);
    }
}

// Handle Image Upload
document.getElementById('imageUpload').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // Display image
    const imagePreview = document.getElementById('imagePreview');
    imagePreview.src = URL.createObjectURL(file);
    imagePreview.style.display = 'block';

    // Wait for image to load to predict
    imagePreview.onload = async () => {
        await predict(imagePreview);
    };
});

async function predict(imageElement) {
    if (!model) return;

    // Preprocessing (Resize and normalize based on your model's needs)
    let tensor = tf.browser.fromPixels(imageElement)
        .resizeNearestNeighbor([224, 224]) // Resize to model input
        .toFloat()
        .expandDims();

    // Prediction
    const prediction = await model.predict(tensor).data();
    document.getElementById('result').innerText = `Prediction: ${prediction}`;
    console.log(prediction);
}

// Initialize
loadModel();
