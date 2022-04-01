import os, sys, yaml
from pathlib import Path
from typing import Union


'''
This module is use to load any yaml file into python dictionary

Author: Lim Zhi Hao		Date: 2018-06-07
Version: st-sensemaking-d2

'''

def replace_keyword(config_dict : dict, replace_dict : dict):


	def check_and_replace(in_str : str, replace_dict : dict):
		assert isinstance(in_str, str)
	
		for keyword, to_replace in replace_dict.items():
			if keyword in in_str:
				in_str = in_str.replace(keyword, to_replace)
			else:	
				in_str = in_str
	
		return in_str



	for config_key, val_line in config_dict.items():

		if isinstance(val_line, dict):
			for key, val_str in val_line.items():
				if isinstance(val_str, str):
					val_str = check_and_replace(val_str, 
								replace_dict)

					val_line[key] = val_str

		elif isinstance(val_line, str):
			val_line = check_and_replace(val_line,
						replace_dict)

			config_dict[config_key] = val_line

	return config_dict



def load_yaml_with_keyword(yamldict : dict, keywords : str = 'keywords'):
	
	# yamlfile: Path to yaml file
	# keyword: Default keyword

	err_msg = 'Please ensure you have specified your keywords in yaml file'
	assert keywords in yamldict.keys(), err_msg
	
	replace_dict = yamldict[keywords]
	for key, value in yamldict.items():
		if key != keywords and isinstance(value, dict):	
			new_value = replace_keyword(value, replace_dict) 
			yamldict[key] = new_value	

	return yamldict 



def load_yaml_no_keyword(yamlfile):

	with open(yamlfile, "r") as stream:
		try:
			yamldict = yaml.safe_load(stream)
		except yaml.YAMLError as exc:
			print(exc)
			yamldict = None

	return yamldict


def parse_yaml(yamlfile, keywords : str = None):

	yamldict = load_yaml_no_keyword(yamlfile)
	if keywords:
		print("Loading keywords: '{}'".format(keywords))
		yamldict = load_yaml_with_keyword(yamldict, keywords)

	return yamldict


def main():

	yamlfile = sys.argv[1]
	keywords = sys.argv[2]
	config = parse_yaml(yamlfile, keywords)

	print(config)



if __name__ == '__main__':

	main()
