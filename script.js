let model;

// 1. Load the model from the local folder
async function loadModel() {
    const status = document.getElementById('result');
    status.innerText = "Loading model...";
    // Point to the relative path of your model.json
    model = await tf.loadLayersModel('tm-model/model.json'); 
    status.innerText = "Model loaded! Ready for upload.";
}

// 2. Handle image upload and prediction
document.getElementById('imageUpload').addEventListener('change', async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    // Show preview
    const imgElement = document.getElementById('imagePreview');
    imgElement.src = URL.createObjectURL(file);
    imgElement.style.display = 'block';

    imgElement.onload = async () => {
        const status = document.getElementById('result');
        status.innerText = "Classifying...";

        // 3. Preprocess and Predict
        const tensor = tf.browser.fromPixels(imgElement)
            .resizeNearestNeighbor([224, 224]) // Adjust size to match your model
            .toFloat()
            .expandDims();

        const predictions = await model.predict(tensor).data();
        
        // 4. Display results (Assumes a classification model)
        status.innerText = `Prediction (Normal , Defective): ${predictions}`; 
    };
});

loadModel();
