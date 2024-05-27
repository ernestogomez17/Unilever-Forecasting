data_path='data/Rainfall/FullWeatherData_Toronto.csv' #path to the MTS data
#cycle=$(50) #12 samples an hour, 24 hour a day
data_root='data/Rainfall' #Directory to the MTS data
#preparing dataset stamp
python ./data/data_process.py gen_stamp --data_path=$data_path --cycle=50 --data_root=$data_root