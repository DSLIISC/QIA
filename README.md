
## Installation

Install Anaconda and create the virtual environment using the environment file: 
```
conda env create -f env.yml 
conda activate qia
```

## For Experiments
> [!NOTE] 
> Please note that, given the inherent probabilistic nature of quantum computing, it is possible that a given experiment may not give
identical results across individual invocations.<br>
> Also, depending on the platform, some experiments may take a few hours to finish.

To Run the SQIA Experiments, execute the following command:
```
python GS_Paper-All_Experiments.py -pid <5/6/7/72> -bp '<result_location>' --no-noisy -r '<repeat the experiment>' -sp '<delta>'
```
You can also run without the repeat and success probability parameter. They will take the default values
```
python GS_Paper-All_Experiments.py -pid <5/6/7/72> -bp '<result_location>' --no-noisy
```
To Run the OQIA Experiments, execute the following command:
```
python QAOA_Paper-All_Experiments.py -pid <5/6/7/71> -bp '<result_location>' --no-noisy --inc -r '<repeat the experiment>' -p '<repetation depth of QAOA>'
```
You can also run without the repeat and success probability parameter. They will take the default values
```
python QAOA_Paper-All_Experiments.py -pid <5/6/7/71> -bp '<result_location>' --no-noisy --inc
```
Note : Repeat is optional. It will take the default value of 10.

### Experiment Log 
Code execution log files are stored in the 'logs' folder 
