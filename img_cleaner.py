# from os import listdir
# from os.path import isfile, join
# onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

import os
from PIL import Image

#import subprocess


passed = 0
invalid = 0

def is_grey_scale(img_path):
    im = Image.open(img_path).convert('RGB')
    w,h = im.size
    for i in range(w):
        for j in range(h):
            r,g,b = im.getpixel((i,j))
            if r != g != b: return False
    return True


main_folder = 'more_breeds/'

directories = [f for f in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, f))]
try: directories.remove('.DS_S')
except: pass

for dir_ in directories:
	for pic in os.listdir(main_folder+dir_):
		pic_path = main_folder+dir_+'/' + pic


		try:
		    if is_grey_scale(pic_path) == True:
		    	os.remove(pic_path)
		    	invalid += 1
		    passed += 1


		except IOError:
			os.remove(pic_path)
			invalid += 1 

			print pic_path, '\n'

		print 'Passed: {}, Invalid: {}'.format(passed, invalid)

		