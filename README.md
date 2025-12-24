# Fine-Grained Fruit Classification

A comprehensive deep learning project for fine-grained classification of fruit cultivars using convolutional neural networks, deployed on a robotic system.

## Overview

This project addresses the challenging task of fine-grained fruit classification, where the goal is to distinguish between closely related cultivars within the same fruit species. Unlike coarse-grained classification (e.g., apple vs. banana), fine-grained classification requires models to learn subtle visual differences such as color gradients, texture patterns, and shape variations that differentiate cultivars like Fuji apples from Jazz apples.

The project follows a phased approach, progressing from dataset collection and baseline model evaluation through fine-tuning and finally to real-world deployment on a robotic platform.

## Dataset

The dataset consists of **21 fruit cultivars** collected under real-world conditions:

- **Apples (8 classes)**: Granny Smith, Pink Lady, Royal Gala, Fuji, Jazz, Kanzi, Modi, SnapDragon
- **Oranges (2 classes)**: Navel, Valencia
- **Grapes (5 classes)**: Thompson Seedless, Crimson Seedless, Autumn Royal, Cotton Candy, Sweet Globe
- **Bananas (2 classes)**: Cavendish, Lady Finger (Baby Banana)
- **Pears (4 classes)**: Beurré Bosc, Nashi, Corella, Red Angel

### Dataset Characteristics

- **Total Images**: ~500-600 images across all classes
- **Images per Class**: 15-50 images, depending on availability
- **Collection Methods**: Manual photography (iPhone 11, iPhone 12 Pro, iPhone X) from supermarkets and markets, supplemented with verified online sources (Shutterstock, Alamy, 123RF)
- **Diversity**: Images captured with variations in:
  - Lighting conditions (daylight, artificial, shade)
  - Backgrounds (retail displays, home surfaces, packaging)
  - Viewing angles (top-down, side profile, close-ups)
  - Ripeness stages and color variants

### Dataset Split

- **Training Set**: 60%
- **Validation Set**: 10%
- **Testing Set**: 30%

## Project Phases

### Phase 1-1: Proposal and Dataset Collection
- Research gap analysis and motivation
- Dataset collection strategy
- Label accuracy verification protocols
- Documentation of collection methodology

**Deliverable**: Proposal report (see `reports/phase_1_1_proposal.pdf`)

### Phase 1-2: Pretrained Model Evaluation
Evaluation of two pretrained CNN architectures on the collected dataset:

- **DenseNet-201**: Densely connected convolutional network
- **EfficientNet-B0**: Efficient architecture with compound scaling

**Analysis includes**:
- Classification accuracy, precision, recall, F1-score
- Confusion matrices
- Misclassification patterns
- Model comparison and selection

**Deliverable**: Analysis notebook (see `notebooks/phase_1_2_pretrained_models.ipynb`)

### Phase 2: Fine-Tuning
Fine-tuning of selected pretrained models with:
- Learning rate scheduling strategies
- Layer-wise fine-tuning approaches
- Data augmentation techniques
- Comprehensive hyperparameter tuning

**Deliverable**: Fine-tuning notebook (see `notebooks/phase_2_fine_tuning.ipynb`)

### Phase 3: Deployment and Monitoring
Deployment of the fine-tuned model on a robotic platform:

- **Selected Classes for Deployment**: 3 grape cultivars (Autumn Royal, Crimson Seedless, Thompson Seedless)
- Robot image capture and fine-tuning on new robot-collected images
- Real-time inference implementation
- Robot action triggers based on classification results

**Deliverables**:
- Deployment notebook (see `notebooks/phase_3_deployment_fine_tuning.ipynb`)
- ROS2 packages for inference and image capture (see `src/`)
- Demonstration video (see `assets/robot_demonstration.mp4`)
- Phase 3 report (see `reports/phase_3_report.docx`)

### Phase 4: Final Report and Presentation
- Comprehensive project documentation
- Oral presentation with results analysis
- System architecture overview
- Discussion of challenges and improvements

**Deliverables**:
- Final report (see `reports/phase_4_report.docx`)
- Presentation slides (see `presentations/phase_4_presentation.pptx`)

## Model Architecture

### EfficientNet-B0 (Deployed Model)

For deployment, EfficientNet-B0 was fine-tuned and selected based on its balance of accuracy and efficiency. The model was fine-tuned for 3 classes (Autumn Royal, Crimson Seedless, and Thompson Seedless grapes) for robotic deployment.

**Model Checkpoint**: `models/efficientnet_b0_group_best_no_aug.pth`

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- ROS2 (Humble or later, for deployment)
- CUDA-capable GPU (recommended for training)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fine-grained-fruit-classification.git
cd fine-grained-fruit-classification
```

2. Install Python dependencies:
```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn pillow tqdm
```

3. For ROS2 deployment, build the packages:
```bash
cd src
colcon build --packages-select image_classification capture_topic
source install/setup.bash
```

## Usage

### Training and Evaluation

Open the Jupyter notebooks in the `notebooks/` directory:

- `phase_1_2_pretrained_models.ipynb`: Baseline model evaluation
- `phase_2_fine_tuning.ipynb`: Fine-tuning experiments
- `phase_3_deployment_fine_tuning.ipynb`: Deployment-specific fine-tuning

### ROS2 Deployment

1. **Set model path** (environment variable):
```bash
export FRUIT_CLASSIFIER_MODEL_PATH=/path/to/models/efficientnet_b0_group_best_no_aug.pth
```

2. **Capture images from robot camera**:
```bash
ros2 run capture_topic capture_subscriber -o /path/to/save/images -n 100
```

3. **Run inference node**:
```bash
ros2 run image_classification inference
```

The inference node subscribes to `/depth_cam/rgb/image_raw` and publishes arm control commands to `/ros_robot_controller/bus_servo/set_position` based on detected fruit classes.

## Project Structure

```
fine-grained-fruit-classification/
├── README.md                          # This file
├── notebooks/                         # Jupyter notebooks for each phase
│   ├── phase_1_2_pretrained_models.ipynb
│   ├── phase_2_fine_tuning.ipynb
│   └── phase_3_deployment_fine_tuning.ipynb
├── reports/                           # Project reports
│   ├── phase_1_1_proposal.pdf
│   ├── phase_3_report.docx
│   └── phase_4_report.docx
├── presentations/                     # Presentation slides
│   └── phase_4_presentation.pptx
├── src/                               # ROS2 source packages
│   ├── robot_image_classifier/        # Inference package
│   │   └── image_classification/
│   │       ├── inference.py           # ROS2 inference node
│   │       └── model_wrapper.py       # Model loading wrapper
│   └── capture_robot_image/           # Image capture package
│       └── capture_topic/
│           └── capture_sub.py         # Camera subscriber node
├── models/                            # Trained model checkpoints
│   └── efficientnet_b0_group_best_no_aug.pth
└── assets/                            # Media files
    └── robot_demonstration.mp4        # Robot deployment demo video
```

## Key Results

- Successfully classified 21 fruit cultivars with fine-grained accuracy
- Achieved robust performance on real-world images with diverse conditions
- Deployed model on robotic platform with real-time inference
- Demonstrated effective robot action triggering based on classification

## Challenges Addressed

1. **Fine-Grained Visual Similarity**: Cultivars within the same fruit type share many visual characteristics, requiring models to learn subtle discriminative features.

2. **Intra-Class Variability**: Natural variation in fruit appearance due to ripeness, lighting, and environmental conditions.

3. **Dataset Collection**: Ensuring label accuracy while collecting diverse, real-world images without using existing benchmark datasets.

4. **Model Deployment**: Adapting models trained on initial datasets to robot-captured images with different hardware and conditions.

5. **Real-Time Inference**: Balancing accuracy and efficiency for real-time classification on robotic platforms.

## Technologies Used

- **Deep Learning**: PyTorch, torchvision
- **Models**: DenseNet-201, EfficientNet-B0
- **Computer Vision**: OpenCV, PIL
- **Robotics**: ROS2 (Robot Operating System 2)
- **Data Analysis**: pandas, numpy, matplotlib, seaborn, scikit-learn

## References

This project builds upon research in fine-grained image classification and agricultural computer vision. Key references are included in the Phase 1-1 proposal report.

## License

This project is part of an academic coursework (COMP8430: Advanced Computer Vision and Action). Please refer to the individual reports for detailed methodology and citations.

## Contributors

Group project for COMP8430 - Advanced Computer Vision and Action

---

For detailed methodology, experimental results, and analysis, please refer to the reports and notebooks in their respective directories.
