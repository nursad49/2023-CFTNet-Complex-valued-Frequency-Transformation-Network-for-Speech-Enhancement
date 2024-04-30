# 2023-CFTNet-Complex-valued-Frequency-Transformation-Network-for-Speech-Enhancement


# ******************* CFTNet ******************
 Complex-valued Frequency Transformation Network for Speech Enhancement
 Authors: Nursadul Mamun, John H.L. Hansen “CFTNet: Complex-valued frequency transformation
 network for speech enhancement, INTERSPEECH, Dublin, Ireland, 2023.

# Architecture
The network consists
of three main components: (i) a fully convolutional complex-
valued encoder-decoder network (Cplx-UNet), (ii) complex-
valued SkipBlocks (SB) within the skip connections between
encoder and decoder, and (iii) complex-valued frequency trans-
formation modules. The encoder network is designed using a
series of encoder blocks followed by an FTL module in the al-
ternate layers until the encoder downsamples to a single pixel.
This ensures the decoder uses all spectral and temporal features
learned by the encoder. Each encoder/decoder block is built
upon complex-valued convolution layers to ensure successive
enhancement of both magnitude and phase. A complex convo-
lutional block in the skip connection reduces the semantic gap
between the encoder and decoder blocks and thus guides the
decoder to reconstruct the enhanced output. Although U-Net
can capture contextual correlation for prediction, it yields less
attention toward harmonic correlation. The proposed CFTNet
can capture long-range dependencies and correlations among
harmonics using FTL layers that CNN does not possess due to
its localized receptive fields. Therefore, CFTNet employs an
FTL block along with the encoder layer to exploit harmonic
structures in the frequency components.



                            ![CFTNet_Network_Overview](https://github.com/nursad49/2023-CFTNet-Complex-valued-Frequency-Transformation-Network-for-Speech-Enhancement/assets/45471274/da66e6e2-ba39-434d-8f1e-bdd2cdf4ef2c)




The folder contains:

AudioDataGeneration.py: This generates the noisy audio samples for different SNRs for the Train, Dev, and Test folders. It saves audio and noisy audio files 			in the Database>Original_Samples>Train/Dev/Test folder. Change the name of the noise and SNRs in the function.

Write_scp_files.py: This generates the .scp files for Dataprep.py. Change the noise name as like AudioDataGenerator.py. 

Dataprep.py: This splits all clean and noisy audio files into 4-second chunks and generates training samples to use to train the system. The generated files 	    are saved in the Database>Training_Samples folder

Network.py: This file contains all proposed networks (CFTNet, DCCTN, and DATCFTNET). You can test each network by running this files seperately. 

Train.py: This file contains all files related to train the model. This import training data from Database>Training_Samples folder and related model from the 	Network.py file.

Test.py: This evaluates the network by using the test samples from Database>Original_Samples>Test folder and generate the enhanced files in the same folder.


Dependencies:

dataloader.py: This helps the train.py function to load all training samples (train and Dev) from Database>Training_Samples folder.

modules.py: It contains all the functions required to design the model.

utils.py: It contains all functions required for the whole system.

objective_metrics.py: This contains all objective speech intelligibility and quality metrics, and loss functions.


How to Run:
												
# Part 1: If running for the first time or folders are not available for this database:

Step 1: Execute AudioDataGeneration.py to generate noisy samples corresponding to clean samples for different noise types and SNRs. Ensure you have clean files in Database>Original_Samples>Clean and noise in Database>Original_Samples>Different_Noise folder to generate rthe equired noisy files. Modify AudioDataGeneration.py according to your specifications for noisy samples and make necessary edits. 
   		*If you already have noisy samples for corresponding clean samples, you can skip this step.*

Step 2: Run `Write_scp_files.py` to generate `Train.scp`, `Dev.scp`, and `Test.scp` files. These files will be utilized in the `Dataprep.py` function to segment all training files.
   		*If you already have noisy samples for corresponding clean samples, you can skip this step.*

Step 3: Execute `Dataprep.py` to segment audio files and create the `Database > Training_Samples` folder.
   		*If you already have these files, you can skip this step.*

# Part 2:  Run these steps to train your model every time



Step 4: To train the model run this function in the Python terminal:-

			python3 train.py --model "$model name" --b "$batch size" --e "$Num of epoch" --loss "$Loss Function" --gpu "$GPUs"
			# For Example: python3 train.py --model CFTNet --b 8 --e 50 --loss SISDR+FreqLoss --gpu '0 1'
			# Simple Example: python3 train.py --model CFTNet


This will save a .ckpt file in the Saved_Models>$Model Name folder.
			Repeat step 4 every time to run the desired model.


Step 5: To test the model run this function in the Python terminal:-

			“python3 test.py”

Please change the model name, and model file from the “Saved_Models>$Model Name” folder in the main file of test.py before testing.


