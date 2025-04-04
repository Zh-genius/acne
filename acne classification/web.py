import os
from flask import Flask, render_template, request
import torch
import torchvision.transforms as T
from PIL import Image

# If you have a custom Model class, import it from your definition file:
from model.mutil_cnn import MultiBranchAcneModel, MultiBranchEfficientNetModel  # or your own class

app = Flask(__name__)

# 1. Define and load your model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MultiBranchEfficientNetModel(num_classes=4)
# Assume model weights are stored in ./models/acne_model.pth
model.load_state_dict(torch.load('models/new.pth', map_location=device))
model.eval().to(device)

# 2. Define image preprocessing (example)
transform = T.Compose([
    T.Resize((224, 224)),  # Depending on your model's input size
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet default mean/std
                std=[0.229, 0.224, 0.225])
])

# 3. Prepare a simple acne severity -> treatment suggestion & expected outcome mapping
recommendations = {
    1: "Mild: Use a gentle cleanser + topical medications (e.g., benzoyl peroxide, adapalene), keep skin clean.",
    2: "Moderate: Add combination topical treatments with oral antibiotics (e.g., doxycycline), can be used under doctor's guidance.",
    3: "Moderate to Severe: Combine oral antibiotics with topical retinoids, regular check-ups, and in severe cases, consider oral isotretinoin.",
    4: "Severe: Recommend systemic treatment under specialist supervision (oral isotretinoin, combined with endocrine therapy), strict monitoring of side effects."
}

expected_outcomes = {
    1: "If consistent care is maintained, noticeable improvement can be seen in a few weeks with a low recurrence rate.",
    2: "About 4-8 weeks for significant improvement, medication adherence and facial cleanliness are key.",
    3: "Longer treatment cycle, need regular check-ups and adjustments to medications, improvement is expected but patience is required.",
    4: "Longer course of treatment, close observation of overall health is necessary, most patients can see significant improvement with proper treatment."
}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded files from the form
        front_img = request.files.get('front_img')
        left_img = request.files.get('left_img')
        right_img = request.files.get('right_img')

        if not front_img or not left_img or not right_img:
            # If any image is missing, return a prompt
            return render_template('index.html', error="Please upload three images (front, left, right)!")

        # Read and preprocess images => PIL -> Tensor
        # Note: Convert the file object to PIL Image first, then apply transform
        def preprocess(img_file):
            img_pil = Image.open(img_file).convert('RGB')
            return transform(img_pil).unsqueeze(0)  # [1, C, H, W]

        img_front = preprocess(front_img)
        img_left = preprocess(left_img)
        img_right = preprocess(right_img)

        # Combine into a list to feed into the model
        # If your model forward receives [img_front, img_left, img_right]
        # You need to concatenate the batch dimension or send them separately to the device
        img_front, img_left, img_right = img_front.to(device), img_left.to(device), img_right.to(device)
        imgs = [img_front, img_left, img_right]

        with torch.no_grad():
            logits = model(imgs)  # logits=[B=1, num_classes], alpha to check attention
            # The result is 1×4

        # Get the predicted class (0~3) or (1~4), depending on how your model was trained
        pred_cls = torch.argmax(logits, dim=1).item()

        # If your training used 0,1,2,3 to represent the 4 levels
        # Add 1 here to get the actual acne severity
        severity_level = pred_cls + 1

        # Get treatment suggestion & expected outcome
        recommendation = recommendations.get(severity_level, "No suggestion available")
        expected_result = expected_outcomes.get(severity_level, "No information available")

        # Return the result to the front-end
        return render_template('index.html',
                               severity=severity_level,
                               recommendation=recommendation,
                               expected_result=expected_result)

    # GET method => Show page
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8888)