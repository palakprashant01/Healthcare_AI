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

We will cover the following concepts: Convolutional Neural Networks, Residual Networks, Transfer Learning, ResUnet Models
# Convolutional Neural Networks:
1. The initial CNN layers extract high-level general features. The final couple of CNN layers perform classification on a specific task.
2. Local features scan the image first, searching for simple shapes and patters. The patterns are then picked up by the subsequent layers to form more complex features.

# Residual Networks (ResNets):
1. As CNNs become deeper, the vanishing gradient problem tends to occur, negatively impacting model performance.
2. ResNets work as an efficient solution to solve this problem. They include the "skip connection" feature, enabling the training of multiple layers without the vanishing gradient issue,
3. ResNets also work by including "identity mappings" over the CNNs.
4. ImageNet trains the ResNet deep network, and includes 11 million images across 11,000 categories.

# Transfer Learning:
1. Transfer Knowledge is a machine learning method to transfer information from a pretrained network to a new network and make minor changes to retrain (or repurpose) it and perform newer (and similar) tasks.
Transfer Knowledge is used to accelerate the model training process and save on computationl power and time to train the model.
2. In transfer knowledge, we take the pretrained network and divide the network such that we extract all the pretrained convolutional layers and copy them into a new network (which would parse through new information). We would also add new dense layers into this new network.

Next, we keep the transfered pre-trained convolutional layers frozen and only train the dense layers to perform a new specific task. As a result, we keep building on previous knowledge and tasks and increase intelligence due to transfer learning.

There are two strategies to transfer learning:
1. Strategy 1: We freeze the trained CNN network weights from the first layers and then only train the newly added dense layers with randomly initialized weights.
2. Strategy 2: We initialzie the CNN network with pretrained weights and retrain the network while setting learning rate to be very small. Setting the learning rate to be small will ensure that we do not alter the trained weights too much.

The following are some challenges in applying the transfer learning technique:
1. Transfer learning can negatively affect the model if the features of the old and new tasks are not related, causing negative transfer. As a result, we must have newer but similar tasks.
2. There are limits to the amount of knowledge transfered from one network to another, called transfer bounds. Transfer bounds help us assess the robustness and efficacy of the model. <br />

Out of the two strategies, we will be using Strategy 1 for our project.

# ResUnet Models:
1. ResUnet Models combine the Unet backbone architecture with residual blocks to prevent the vanishing gradient problem generally occuring with deep networks.
2. ResUnet models are based on Full CNNs to perform well on segmentation tasks.
3. The ResUnet architecture consists of 3 components: (1) Encoder or contracting path (2) Bottleneck (3) Decoder or expansive path

Delving deeper into our architecture, we have:
1. Encoder (Contracting Path): This path contains several contraction blocks, which parse through residual blocks, followed by 2x2 max pooling. After each block doubles, the features map, helping the model gain intelligence and learn complex features with ease.
2. Bottleneck: This serves as a connection between the encoder and the decoder. It takes the input and then passes through a residual block and 2x2 up-sampling convolutional layers.
3. Decoder (Expansive Path): Each block in this path takes up-sampled inouts from the previous layer and concatenates them with the outputs from the residual blocks in the encoder. This ensures that features learned in the encoder are utilized while reconstructing the data. The output from the residual block in the final layer passes through a 1x1 convolutional layer and produces the output with the same input size.
