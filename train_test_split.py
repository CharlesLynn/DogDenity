import os
import cv2
import shutil
from random import shuffle

def files_list(dir_):
	#Creates list of files from a directory path.
    file_paths = [dir_+f for f in os.listdir(dir_)]
    try:
        file_paths.remove(dir_+'.DS_Store')
    except:
        pass
    return file_paths

def directory_list(path):
	#Creates list of directories.
	directories = [path+'/'+f+'/' for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
	try:
		directories.remove(path+'/'+'.DS_Store'+'/')
	except:
		pass

	return directories


def clean(image_path):
    #Deletes gif files.
    if image_path.split('.')[-1] == 'gif':
        os.remove(image_path)

    #Tries to open images in opencv, if there is an error the image is deleted.
    else:
        try:
            img = cv2.imread(image_path)
            img = cv2.resize(img, (150, 150), interpolation = cv2.INTER_CUBIC)
            return 0  # #Returns 0 for invalid image count.

        except:
                print image_path
                os.remove(image_path)
                return 1 #Return 1 for invalid image count

   

def image_train_test_split(orgin_dir, new_dir, n):
    image_train_test_split takes

 	"""
	image_train_test_split moves n number of images per class, to a test_split folder.
 	Images should arranged as follows. All destination directories and subdirectories are created.
 	
 	train_split/
 		-class1/
 			-image0.jpg
 			-image1.jpg
 			...
 		-class2/
 			-image0.jpg
 			-image1.jpg
 			...
 		-class3/
 			-image0.jpg
 			-image1.jpg
 			...
 	"""

    #Creates the new test_split directory.
    if not os.path.isdir(new_dir): os.makedirs(new_dir)

	    
    for dir_ in directory_list(orgin_dir):
        if not os.path.exists(dir_.replace(orgin_dir, new_dir)):
            os.makedirs(dir_.replace(orgin_dir, new_dir))

        image_paths = files_list(dir_)
        shuffle(image_paths) #Shuffles for random sample from each class.
        
        #Moves n files to test directory
        for img_path in image_paths[:n]:
            os.rename(img_path, img_path.replace(orgin_dir, new_dir))

if __name__ == '__main__':
    #Test_train_split, moves n number of photos randomly from one folder to another.
    orgin_dir = 'data/trian_split' #Orign folder, training images.
    new_dir = 'data/test_split' #New, test folder.
    n = 200  #Number of images per class for the test_split
    image_train_test_split(orgin_dir, new_dir, n)
    print 'Done'
