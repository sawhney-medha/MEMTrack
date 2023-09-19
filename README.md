[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
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
#### ADding Data

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
   - RUn notebook

6. **Preprocessed Data Directory:**
   - Expected Directory Structure after preprocessing code:
   
     ```
     ├── MEMTRack/
     │   ├──data/
     │   │   ├── videomap.txt
     │   │   ├── collagen/
     |   |   |     ├── video55/
     |   |   |     |    ├── frame1/
     |   |   |     |    |    ├── annotations_motility_no/
     |   |   |     |    |    |    ├── 0.txt
     |   |   |     |    |    |    ├── 1.txt
     |   |   |     |    |    |    └── ...
     |   |   |     |    |    ├── annotations_motility_low/
     |   |   |     |    |    ├── annotations_motility_mid/
     |   |   |     |    |    ├── annotations_motility_high/
     |   |   |     |    |    ├── bacteria/
     |   |   |     |    |    |    ├── 1/
     |   |   |     |    |    |    |    ├── xy_coord/
     |   |   |     |    |    |    |          ├── 0.txt
     |   |   |     |    |    |    |          ├── 1.txt
     |   |   |     |    |    |    |          └── ...
     |   |   |     |    |    |    |
     |   |   |     |    |    |    ├── 2/
     |   |   |     |    |    |    └── ...
     |   |   |     |    |    ├── images/
     |   |   |     |    |         ├── 0.tif
     |   |   |     |    |         ├── 1.tif
     |   |   |     |    |         └── ...
     |   |   |     └── ...
     │   │   └── ...
     │   ├── Raw Data.csv
     ```
  

### Data Usage

7. **Feature Generation Code:**
   - Run notebook

6. **Final Data Directory:**
   - Expected Directory Structure after preprocessing code:
   - 
7. **Data Loading:**
   - Show how to load the preprocessed data within your codebase. Provide code snippets that demonstrate how to access and use the data for training, testing, or other tasks.

8. **Data Splitting:**
   - If relevant, explain how to split the data into training, validation, and testing sets.





### Acknowledgement
This research was supported in part by NSF grants CBET-2133739, CBET-1454226 and 4-VA grant to [Dr. Bahareh Behkam](https://me.vt.edu/people/faculty/behkam-bahareh.html), and NSF grant IIS-2107332 to [Dr. Anuj Karpatne](https://people.cs.vt.edu/karpatne/). Access to computing resources was provided by the Advanced Research Computing (ARC) Center at Virginia Tech.
