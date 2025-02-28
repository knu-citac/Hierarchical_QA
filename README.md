# Hierarchical_QA

# ROS2 Package for Hierarchical QA-Based Scene Understanding

This repository contains a ROS2 package for real-time scene understanding in autonomous vehicles. The system employs a hierarchical question-answering (QA) approach integrated with a Vision-Language Model (VLM) to generate structured scene descriptions.

## Installation

### Prerequisites
Ensure you have ROS2 installed on your system. This package has been tested on ROS2 Humble.

### Additional Steps for VLM
To run the Vision-Language Model (VLM), install the required Python dependencies:

```sh
pip install torch
pip install transformers
pip install pillow
```

### Model Weights
The weights of the finetuned Vision-Language Model (VLM) must be placed in the following directory:

```
./src/hierarchical_qa/hierarchical_qa/total92.pth
```


