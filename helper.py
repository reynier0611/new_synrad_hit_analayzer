import os

def create_directories(detectors):
	dirnames = ['output_plots','processed_data']

	for dirname in dirnames:
		if not os.path.exists(dirname):
   			os.makedirs(dirname)

		for detector in detectors:
			new_dirname = os.path.join(dirname,detector)
			if not os.path.exists(new_dirname):
   				os.makedirs(new_dirname)
