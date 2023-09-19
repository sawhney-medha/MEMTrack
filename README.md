[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) ![Python Version](https://img.shields.io/badge/python-3.8-blue) ![PyTorch Version](https://img.shields.io/badge/pytorch-1.9.0-blue) ![Detectron2 Version](https://img.shields.io/badge/Detectron2-v0.5.1-blue)




# MEMTrack
Deep learning based automated detection and tracking of bacteria in complex environments such as Collagen.

## Repository Status
🚧 This repository is currently undergoing updates and is a work in progress. 🚧

We appreciate your patience and understanding. Feel free to check back later for the latest updates and improvements.

## Project Description
Tracking microrobots is a challenging task, considering their minute size and high speed. As the field progresses towards developing microrobots for biomedical applications and studying them in physiologically relevant or in vivo environments, this challenge is exacerbated by the dense surrounding environments with feature size and shape comparable to those of microrobots. To address this challenge, we developed Motion Enhanced Multi-level Tracker (MEMTrack), a robust pipeline for detecting and tracking micro-motors in bright-field microscopy videos using synthetic motion features, deep learning-based object detection, and a modified Simple Online and Real-time Tracking (SORT) algorithm with interpolation for tracking. Our object detection approach combines different models based on the object’s motion pattern. We trained and validated our model using bacterial micro- motors in the tissue-like collagen environment and tested it in collagen and liquid (aqueous) media. We demonstrate that MEMTrack can accurately predict and track even the most challenging bacterial micro-motors missed by skilled human annotators, achieving precision and recall of 77% and 48% in collagen and 94% and 35% in liquid media, respectively. We also show that MEMTrack is able to accurately quantitate the average speed of bacterial micromotors with no statistically significant difference from the laboriously produced manual tracking data. Our proposed pipeline not only represents a significant contribution to the field of microrobot image analysis and tracking using computer vision but also opens the potential of applying deep learning methods in vision-based control of microrobots for various applications, including disease diagnosis and treatment.

## Getting Started
### Installation
To get started with this project, follow these steps:

1. Clone the repository to your local machine using Git:

   ```bash
   git clone https://github.com/sawhney-medha/MEMTrack.git
   
2. Navigate to the project directory:

   ```bash
   cd MEMTrack

3. Install the required dependencies using pip:

   ```bash
    pip install -r requirements.txt
4. Download models.zip containing pre-trained and store it in *MEMTrack/models*

**[Download Pretrained Models](https://drive.google.com/file/d/1agsLD5HV_VmDNpDhjHXTCAVmGUm2IQ6p/view?usp=sharing)**

### Data Preparation
#### Adding Data

1. **Data Collection**
   -  We recorded bacteria (i.e., the micromotor in bacteria-based biohybrid microrobots) swimming behavior in collagen, as a tissue surrogate, and in an aqueous environment. In order to generate training, validation, and test datasets for MEMTrack, the microscopy videos acquired in collagen and in aqueous media were imported into ImageJ software. MTrackJ plugin was used to label all bacteria in each frame of each video manually, and their x and y coordinates were recorded.
   -  Any data can be used as long as a list of frames along with their respective x and y coordinates are available.
   -  To evaluate the results, track_ids would also be required for the annotated bacteria.
   -  To be able to train different models based on different motilites, motility sub population labels would also be required. If these are not available, one can train a single model for all annotated bacteria.
     
2. **Organize Data Directory:**
   - The final data should be packed in a zip file with the following structure: 
     ```
     ├── video_zip_file/
     │   ├── Images without Labels/
     │   │   ├── sample_name1.jpg
     │   │   ├── sample_name2.jpg
     │   │   └── ...
     │   ├── Raw Data.csv
     ```

3. **Data Format:**
  - RawData.csv should have the following columns:
      - "TID" --> Track Id
      - "PID" --> Picture Id/Frame Number
      - "x [pixel]"" -->  x coordinate
      - "y [pixel]"  --> y coordinate
      - "subpopulation" -->containing "N"/"L"/"M"/"H" for subpopulations (Non Motile/Low Motility/Mid Motility/HIgh Motility)

#### Preprocessing Data


4. **Preprocessing Code:**
   - Run the DataPreparation_Motility.ipynb notebook located in the *MEMTrack/src* directory and update the path variables (Cell 3) according to your directory structure.
   - The code needs to be run for every video that needs to be loaded to *data/*
   - The videomap.txt will automatically be generated after the preprocessing code

6. **Preprocessed Data Directory:**
   - Expected Directory Structure after running the preprocessing code:
   
     ```
     ├── MEMTRack/
     │   ├──data/
     │   │   ├── videomap.txt
     │   │   ├── collagen/
     |   |   |     ├── video1/
     |   |   |     |    ├── frame1/
     |   |   |     |         ├── annotations_motility_no/
     |   |   |     |         |    ├── 0.txt
     |   |   |     |         |    ├── 1.txt
     |   |   |     |         |    └── ...
     |   |   |     |         ├── annotations_motility_low/
     |   |   |     |         ├── annotations_motility_mid/
     |   |   |     |         ├── annotations_motility_high/
     |   |   |     |         ├── bacteria/
     |   |   |     |         |    ├── 1/
     |   |   |     |         |    |    ├── xy_coord/
     |   |   |     |         |    |          ├── 0.txt
     |   |   |     |         |    |          ├── 1.txt
     |   |   |     |         |    |          └── ...
     |   |   |     |         |    |
     |   |   |     |         |    ├── 2/
     |   |   |     |         |    └── ...
     |   |   |     |         ├── images/
     |   |   |     |              ├── 0.tif
     |   |   |     |              ├── 1.tif
     |   |   |     |              └── ...
     |   |   |     ├── video2/
     |   |   |     └── ...
     │   │   └── ...
     │   ├── src/
     ```
  

#### Data Usage

7. **Feature Generation Code:**
   - Run the DataFeaturePreparation.ipynb notebook located in the *MEMTrack/src* directory and update the path variables (Cell 3) according to your directory structure.
   - Also update path variables and choose feature generation method at the end of the notebook to generate and store features that would be used for training.
   - The notebook provides multiple ways to generate features, the one recommended based on experiments on Collagen data is: *"optical_flow_median_back"*. This generates 3 channels for each frame: 1. Original Frame, 2. Consecutive OPtical Flow Vector and 3. Difference from Median Background.
   - Similarly, *"optical flow median back"* with *"optical_flow_prior"=x* variable is optical flow from xth previous frame. *"diff_from_max_absolute_consecutive_frame_diff"* creates a feature for difference from the "max consecutive frame diff" feature, with a frame diff prior for xth frame diff
   - The train/test/val split can be provided in this code as dict of video numbers that have been loaded and accordingly their fearture sets will be generated.


6. **Final Data Directory:**
   - Expected Directory Structure after feature generation code:
  
     ```
        ├── MEMTRack/
        │   ├──data/
        │   ├──data_features/
        │   │   ├── exp_name/
        |   |   |     ├── data_features_set/
        |   |   |     |    ├── train/
        |   |   |     |    |     ├── annotations_motility_no/
        |   |   |     |    |     |    ├── 0.txt
        |   |   |     |    |     |    ├── 1.txt
        |   |   |     |    |     |    └── ...
        |   |   |     |    |     ├── annotations_motility_low/
        |   |   |     |    |     ├── annotations_motility_mid/
        |   |   |     |    |     ├── annotations_motility_high/
        |   |   |     |    |     ├── images/
        |   |   |     |    |          ├── 0.tif
        |   |   |     |    |          ├── 1.tif
        |   |   |     |    |          └── ...
        |   |   |     |    |     ├── images_feature/
        |   |   |     |    |          ├── 0.tif
        |   |   |     |    |          ├── 1.tif
        |   |   |     |    |          └── ...
        |   |   |     |    ├── test/
        |   |   |     |    ├── val/
        |   |   |     ├── data_feature_video1/test/
        |   |   |     ├── data_feature_video2/test/
        |   |   |     └── ...
        │   │   └── ...
        │   ├── src/

### Running the Code
The following section describes the training, inference, tracking and evaluation procedures. The codebase is built using PYthon, PyTorch and Detectron 2.0.

#### Training
- Run the training script */scripts/train.sh* to start training from the *MEMTRack/* root directory.
- Update *exp_name*, *data_path* (feature directory) and *output_dir* paths as approriate.
- The training parameters such as learning rate, epochs, etc. can be updated from the bash script.
- There are two python scripts for training: *src/trainBacteriaRetinanetMotionData_Motility.py* and *src/trainBacteriaRetinanetMotionData_Motility_Val_loss.py*. The only difference is that the latter generates loss plots for validation set during training that the Detectron2 code does not automatically generate, which can be visualized in the *src/Train_Val_Loss_Curve.ipynb* notebook. The code for it has been taken from the following repository: https://gist.github.com/ortegatron/c0dad15e49c2b74de8bb09a5615d9f6b
- The training script saves regular checkpoints along with the final model, which are stored in the *output_dir* specifed in the bash script.

  ```bash
   bash scripts/train.sh

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgement
This research was supported in part by NSF grants CBET-2133739, CBET-1454226 and 4-VA grant to [Dr. Bahareh Behkam](https://me.vt.edu/people/faculty/behkam-bahareh.html), and NSF grant IIS-2107332 to [Dr. Anuj Karpatne](https://people.cs.vt.edu/karpatne/). Access to computing resources was provided by the Advanced Research Computing (ARC) Center at Virginia Tech.
