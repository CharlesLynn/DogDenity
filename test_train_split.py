import os
import cv2
import shutil
from random import shuffle
removed=0

def files_list(dir_):
    file_paths = [dir_+f for f in os.listdir(dir_)]
    try:
        file_paths.remove(dir_+'.DS_Store')
    except:
        pass
    return file_paths

def directory_list(path):
	#Creates list of directories for chosen number of catigories.
	directories = [path+'/'+f+'/' for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
	try:
		directories.remove(path+'/'+'.DS_Store'+'/')
	except:
		pass

	return directories


def clean(image_path):
    removed = 0
    np_img = None
    img = None
    if image_path.split('.')[-1] == 'gif':
        os.remove(image_path)
    elif image_path.split('/')[-1] == 'invalid':
        shutil.rmtree(image_path)
    else:
        #print image_path
        try:
            img = cv2.imread(image_path)
            img = cv2.resize(img, (150, 150), interpolation = cv2.INTER_CUBIC)

        except:
                print image_path
                os.remove(image_path)
                removed +=1


    return removed

def image_train_test_split(orgin_dir, new_dir, n):
    if not os.path.isdir(new_dir): os.makedirs(new_dir)

    for dir_ in directory_list(orgin_dir):
        if not os.path.exists(dir_.replace(orgin_dir, new_dir)):
            os.makedirs(dir_.replace(orgin_dir, new_dir))

        image_paths = files_list(dir_)
        shuffle(image_paths)
        for img_path in image_paths[:n]:
            os.rename(img_path, img_path.replace(orgin_dir, new_dir))

if __name__ == '__main__':
    #Test_train_split, moves n number of photos randomly from one folder to another.
    # orgin_dir = 'data/5_split'
    # new_dir = 'data/55_split'
    # n = 5
    # image_train_test_split(orgin_dir, new_dir, n)
    # # print 'done!'
