import os, sys
from pathlib import Path
from importlib.machinery import SourceFileLoader
from importlib import import_module 


'''  
This is to load modules from python dynamically	

For example:

	# Normally, we would do this:
	-----------------------------------------------------------------------

	from numpy.random import randint
	
	low, high = 10, 100
	size = (2, 3)
	
	x = randint(low, high, size)	

	# This example allows us to randomly generate a 2X3 array of integers
	# However, we can also do this dynamically via python's importlib 
	-----------------------------------------------------------------------
	
	# Suppose we have a config (dictionary of dictionaries):
	
	config['transform'] = {'module_name': 'numpy.random',
				'class_name': 'randint',
				'kwargs':{'low': 10, 'high': 100, 'size':[2, 3]}

	# We pass the dictionary as variables as follows:
	-----------------------------------------------------------------------

	from importlib import import_module

	module_name = 'numpy.random'
	class_name = 'randint'
	kwargs = {'low': 10, 'high': 100, 'size':[2, 3]}

	loaded_module = import_module(module_name)
	imported_class = getattr(loaded_module, class_name) 
	x = imported_class(**kwargs)
	
	-----------------------------------------------------------------------	

	# Both approaches allows us to initialize and generate a random array x
	# However, the second approach allows a more dynamic way to load ANY 
	# python function. 


Author: Lim Zhi Hao		Date: 2021-01-03
Version: st-sensemaking-d2

'''

def modules_loader(module_name : str, 
		class_name : str,
		kwargs : dict,
		module_path : Path = None):

	# If no module path is specified, load directly
	if not module_path:
		loaded_module = import_module(module_name)	
	else:
		# Otherwise, load from source file, given the path
		module_file = os.path.join(module_path, module_name)
		loaded_module = SourceFileLoader(class_name, module_file).load_module()

	# Import class from loaded module
	imported_class = getattr(loaded_module, class_name)	

	# Pass arguments into class
	return imported_class(**kwargs)


def main():

	# Specify name of module, class and inputs
	module_path = None
	module_name = 'numpy.random'
	class_name = 'randint'
	kwargs = {'low': 10, 'high': 100, 'size':[2, 3]}

	# Load dynamically and get outputs
	x = modules_loader(module_path = module_path,
			module_name = module_name, 
			class_name = class_name,
			kwargs = kwargs)

	print('An integer array of size: {}'.format(kwargs['size']))	
	print(x)
	

if __name__ == '__main__':

	main()

