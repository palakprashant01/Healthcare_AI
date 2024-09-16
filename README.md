# Healthcare_AI

Purpose of the project:
1. Utilize AI to enhance the accuracy and efficiency of identifying and localizing diseases.
2. Develop a ResUnet segmentation model, training it to detect brain tumors within images.
3. Understand the concept and significance of transfer learning, including the optimal circumstances for its application to speed up model training.
4. Assess the performance of both the ResNet classifier and ResUnet segmentation models using test datasets.
5. Implement the Keras API for the construction of deep convolutional neural networks.
6. Employ Plotly to create dynamic and interactive data visualizations.

AI is revolutionizing various aspects of healthcare, including surgical robotics, diagnosing diseases through medical imaging, and enhancing hospital operations.

Deep learning techniques have shown to be highly effective in disease detection, often outperforming traditional methods like X-rays, MRI, and CT scans in both speed and accuracy.

In our project, we aim to develop a model that can identify and pinpoint brain tumors using MRI scans. This advancement could greatly decrease the expenses associated with cancer diagnosis and facilitate the early detection of tumors.

We have been provided with 3929 brain MRI scans along with their brain tumor locations.

We will perform a two stage detection:
1. Train a ResNet deep learning classifier model to detect whether a tumor exists in the image or not.
2. If a tumor has been detected, we will train a ResUnet segmentation model to localize the tumor location on the pixel level.

For this project, we will also perform image segmentation (using the ResUnet architecture) for the following purposes:
1. Understand and extract information from images at the pixel-level.
2. Train a neural network to produce a pixel-wise mask of the image.
3. Perform object recognition and localization. <br />

In the Unet architecture, we encode the image into a vector and decode it back into an image with the same size. Unet formulates a loss function for every pizel in the input image.

A softmax function is applied to every pixel, helping the segmentation problem work as a classification problem, wherein every pixel of the image undergoes classification.
