import os
import urllib
from PIL import Image
import subprocess
import multiprocessing

n_urls = 0
downloaded = 0
error_count = 0
downloaded = 0.0
skipped = 0



def get_filepaths(urlfiles_path):
	return [urlfiles_path + f for f in os.listdir(urlfiles_path) if os.path.isfile(os.path.join(urlfiles_path, f))][1:]

def file_2_lineslist(path):
	with open(path) as urls:
		return [x.replace('\r\n', '') for x in urls]

def scrape(url_folder, picture_folder):

	if not os.path.exists(picture_folder): os.makedirs(picture_folder)
	global error_count
	global skipped
	global downloaded
	global n_urls

	#For gobal %
	for path in get_filepaths(url_folder):
		n_urls += len(file_2_lineslist(path))

	for path in get_filepaths(url_folder):

		

		for url in file_2_lineslist(path):
			
			#Creates Diectory for each txt file.
			
			name = path.split('/')[-1][:-4]

			if not os.path.exists(picture_folder+name+'/'): os.makedirs(picture_folder+name +"/")		
			
			#Downloads pictures into folders if pic does not exist.
			full_path = picture_folder + name + '/' +  url.split('/')[-1]
			if not os.path.exists(full_path):
				try:
					urllib.urlretrieve(url, full_path)
					downloaded +=1
					print 'Complete: %{}, Downloaded: {}, Errors: {}'.format(round(downloaded*100/n_urls, 1), downloaded, error_count)
				
				except:
					error_count +=1
					print 'Error 404: Missing url {} continuing...'.format(error_count)
					print 'Complete: %{}, Downloaded: {}, Errors: {}'.format(round(downloaded*100/n_urls, 1), downloaded, error_count)
				#print full_path


    		else: 
    			skipped += 1



if __name__ == '__main__':
	scrape('./urls1/', './more_breeds/')




