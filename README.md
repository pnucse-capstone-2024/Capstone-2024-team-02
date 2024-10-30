### 1. Project Introduction

#### 1.1 Background and Necessity
Cardiac magnetic resonance (cMRI) is widely recognized as a key tool in diagnosing and assessing cardiac diseases. Essential information about the left ventricle (LV), myocardium, and right ventricle (RV) is obtained through cMRI, guiding disease classification and treatment. However, manual segmentation, commonly used in clinical settings, is time-intensive and prone to errors. As a solution, deep learning-based techniques have emerged, offering improved accuracy and efficiency, yet they face challenges in generalizing across diverse datasets and clinical conditions.

#### 1.2 Objectives and Key Content
Our project aims to address the limitations of current cardiac segmentation methods by developing a deep learning model optimized for generalization across multi-center and multi-vendor datasets. Key objectives include:
1. Preprocessing multi-center, multi-vendor datasets to prepare for model training.
2. Visualizing cMRI images and segmentation labels for dataset analysis.
3. Developing a deep learning model for robust cardiac segmentation.
4. Training the model to enhance adaptability across diverse imaging conditions.
5. Evaluating model performance on generalization to various clinical datasets.
6. Applying domain adaptation and data augmentation to improve generalizability.
7. Promoting the practical application of the model in clinical settings to streamline cardiac diagnosis.

## 2. Detailed Design
### 2.1 System Architecture

The architecture for the cardiac MRI segmentation project includes the following main components:

- **Front-end**: Provides a web interface for uploading MRI scans, visualizing segmentation results, and managing user settings. This interface includes options for users to view dynamic heart anatomy in different cardiac phases, such as end-diastole (ED) and end-systole (ES), with views in both long-axis (LA) and short-axis (SA) orientations.

- **Back-end**: A server that processes cardiac MRI data and performs model-based segmentation of heart structures. The server is responsible for handling requests, initiating segmentation tasks using the U-Net model, and generating animations for cine MRI visualization. The back-end also manages user data and tracks segmentation progress.

- **Database**: A secure, cloud-based database that stores patient MRI data, segmentation results, user-uploaded scans, and data related to different cardiac pathologies. It includes annotated segmentation labels (for structures like the left ventricle, right ventricle, and myocardium) that are crucial for training and evaluating model performance. Additionally, the database tracks metadata for each MRI scan, such as vendor-specific attributes, scan protocols, and patient demographics.

This architecture is optimized to handle large datasets, multi-center imaging variability, and real-time user interactions, ensuring that the platform is both robust and scalable for research and clinical applications.


## 3. Installation and Usage Instructions
### Installation Requirements
- **Software**:
  - Python 3.8 or higher
  - Flask (for web application development)
  - VTK (for 3D rendering of heart structures)
  - Nibabel (for handling NIfTI MRI files)
  - OpenCV (for generating MRI animations)
  - PyTorch (for model-based segmentation tasks)
  - AWS CLI (if deploying to cloud infrastructure)

### Installation Steps
1. **Clone the repository**:
   ```bash
   git clone https://github.com/pnucse-capstone-2024/Capstone-2024-team-02.git
   ```

2. **Download and place the pre-trained models**:
   - Download the model files from [Google Drive](https://drive.google.com/file/d/1nzAk1xfFLaEDYO809oWYlEEtEE9z1PaK/view?usp=sharing).
   - Move the downloaded models to the `models` directory in the project folder.

3. **Install required libraries**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure the database** (if using a cloud-based database):
   - Set up the database and update credentials in `config.py`.

5. **Run the Flask application**:
   ```bash
   flask run
   ```

6. **Access the web interface**:
   - Open your browser and go to `http://localhost:5000`.

### Usage
1. Access the platform by logging in to your user account.
2. Upload a cardiac MRI scan (in `.nii` or `.nii.gz` format).
3. Choose MRI scan options (e.g., 'cine', 'ed', 'es') and orientation ('SA' or 'LA').
4. Start the segmentation task, and view segmentation results and dynamic heart visualizations in different cardiac phases.
5. Save or download segmentation results and animations as needed.

## 4. Introduction and Demo Video
Watch the project introduction and demo video, which demonstrates MRI segmentation, 3D visualizations, and how users can interact with the heart anatomy viewer.


[![Project Introduction](https://img.youtube.com/vi/_MsC9S7zIS4/0.jpg)](https://youtu.be/_MsC9S7zIS4?si=3L1yC1ygt70XV9im)

## 5. Team Introduction

| **Name**                   | **Role**                       | **Key Responsibilities**                                                                                                                                              |
|----------------------------|--------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Islam Salikh (이슬람 살리흐)**     | Data Engineer & Backend Dev    | - Preprocessed and visualized MRI data for training. <br> - Implemented post-processing for cleaner segmentation output. <br> - Integrated the model in Flask for real-time interaction. |
| **Kenes Yerassyl (케네스 예라슬)** | Machine Learning Engineer      | - Designed and trained the U-Net model, ensuring generalization. <br> - Handled model integration and security features for secure file handling and account management. |
| **Nugayeva Altynay (누가예바 알트나이)** | Data Scientist & Frontend Dev  | - Prepared and augmented MRI data. <br> - Evaluated model performance with benchmark metrics. <br> - Developed the front-end interface for file uploads, results viewing, and profiles. |


