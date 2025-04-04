# AI-based framework for acne grading and treatment outcome prediction
## Project Overview
This project aims to use artificial intelligence technology to develop a framework that can automatically grade the severity of acne and predict treatment outcomes, assist clinical decision-making, and improve the accuracy and personalization of acne treatment.

## Background Introduction
Acne is a common skin disease that affects about 85% of adolescents and young people worldwide. It not only affects the appearance, but also has a negative impact on the patient's psychology. At present, the assessment of acne severity mainly relies on the subjective visual judgment of clinicians, which is time-consuming and susceptible to observer differences. At the same time, existing AI research focuses on acne severity grading, and pays less attention to the prediction of treatment outcomes. However, acne treatment plans are complex and individual efficacy varies greatly. There is an urgent need for automated tools that can comprehensively evaluate the condition and predict treatment responses.

## Technical Solution
### Acne Severity Grading
1. **Data Collection**: 2706 clinical acne images of 902 patients were collected from the Third People's Hospital of Zhuhai City, Guangdong Province, China. Each patient included standardized images from three perspectives: left face, front face, and right face. All images were collected under consistent clinical conditions to ensure high data quality and repeatability.
2. **Data annotation**: Experienced dermatologists annotated the severity of acne according to the 2019 Chinese Guidelines for the Management of Acne Vulgaris, which was divided into four levels: mild (only acne), moderate (inflammatory papules), moderate (pustule), and severe (nodules and cysts). The annotation process was independently performed by at least two certified dermatologists, and disagreements were resolved through consensus discussion.
3. **Model architecture**: A multi-branch convolutional neural network (CNN) architecture was used. Each branch used advanced backbone networks such as ResNet, VGG, or EfficientNet to extract features from a single perspective, and then fused the features through a connection-based fusion network, and then classified the severity of acne through a fully connected layer and a softmax activation function.

### Treatment outcome prediction
1. **Data collection**: Clinical medication data of 248 acne patients with a treatment duration of less than 8 weeks were obtained from the same hospital, including personal demographic information, detailed prescription records, and acne severity levels before and after treatment. Personal identifiable information was removed, and demographic variables related to model training (gender and age) were retained.
2. **Data processing**: The drug types were standardized and classified according to the Chinese Guidelines for Diagnosis and Treatment of Acne (2019 Revised Edition), and derived features such as the total number of treatment types and treatment duration (weeks) were created. The prediction target is a binary classification label, 1 indicates that the severity of acne has improved after treatment, and 0 indicates no improvement.
3. **Model selection**: Multiple machine learning algorithms such as random forest, support vector machine (SVM), logistic regression, K nearest neighbor (KNN), and multilayer perceptron (MLP) were used for evaluation. Finally, CatBoost classifier was selected as the best model, with an accuracy of 0.833 under 10-fold cross validation, and high precision (0.862) and F1 score (0.709).

## Experimental results
### Acne severity classification
Among all the tested architectures, EfficientNet_b0 performed best with an accuracy of 0.930, precision, recall, and F1 scores of 0.912. Other EfficientNet variants such as EfficientNet_b3 (accuracy 0.921) and EfficientNet_b1 (accuracy 0.904) also performed well. In contrast, traditional models such as ResNet (accuracy 0.877) and VGG (accuracy 0.895) performed slightly worse, while YOLOv8, as an object detection algorithm, performed poorly on the acne severity classification task with an accuracy of only 0.367 and an F1 score of 0.295.

### Treatment outcome prediction
CatBoost classifier performed best in treatment outcome prediction. SHAP analysis found that the severity of acne before treatment was the most important factor affecting the model prediction, and the duration of treatment and the total number of treatment types also had an important impact. Factors such as age, gender, and specific drug types have relatively small effects, but they are still significant.

## Project Significance
The framework developed in this project effectively combines deep learning and machine learning techniques, and has achieved good results in acne severity classification and treatment outcome prediction, providing clinicians with data-driven decision support, which helps to achieve personalized and precise acne treatment.

## Contribution Guide
Interested developers are welcome to participate in project contributions, such as optimizing model performance, improving data processing methods, and improving project documentation. Please read the project's contribution guide before contributing to understand the code specifications and submission process.

## Contact Information
If you have any questions or suggestions, please contact the project leader by email: shuli@mpu.edu.mo
