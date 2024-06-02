
import glob

import cv2
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def extract_lip_embeddings(data_dir, input_file):
	import face_recognition
	video_path = data_dir + '/' + input_file
	identifier = video_path.split('/')[-1].split('.')[0]  # This will be the video identifier without file extension
	video_capture = cv2.VideoCapture(video_path)

	all_frame_data = ""

	frame_number = 0
	while video_capture.isOpened():
		ret, frame = video_capture.read()
		if not ret:
			break

		rgb_frame = frame[:, :, ::-1]
		face_landmarks_list = face_recognition.face_landmarks(rgb_frame)

		for face_landmarks in face_landmarks_list:
			if 'top_lip' in face_landmarks and 'bottom_lip' in face_landmarks:
				top_lip = face_landmarks['top_lip']
				bottom_lip = face_landmarks['bottom_lip']
				all_points = top_lip + bottom_lip
				x_coordinates = [p[0] for p in all_points]
				y_coordinates = [p[1] for p in all_points]
				left, right, top, bottom = min(x_coordinates), max(x_coordinates), min(y_coordinates), max(y_coordinates)

				# Collect coordinates for each frame
				all_frame_data += f"{left}/{right}/{top}/{bottom}/"
			else:
				face_locations = face_recognition.face_locations(rgb_frame)
				if face_locations:
					print("No lips found, so setting to entire face")
					top, right, bottom, left = face_locations[0]
					all_frame_data += f"{left}/{right}/{top}/{bottom}/"
				else:
					# If no face found, set it to the entire frame
					print("No face nor lips found, setting it to the entire frame. This is not good, but its the best i can think of right now")
					height, width, _ = frame.shape
					all_frame_data += f"0/{width}/{0}/{height}/"
				
		frame_number += 1

	video_capture.release()
	return f"{input_file}/{all_frame_data}"

from sklearn.model_selection import train_test_split

def process_file(file, data_dir):
	return extract_lip_embeddings(data_dir, file)

def preprocess_video(data_dir, data_ext, output_dir, outputfilename):
    files = glob.glob(f'{data_dir}/*.{data_ext}', recursive=True)
    files = [os.path.relpath(file, data_dir) for file in files]

    trainval_files = files# test_files = train_test_split(files, test_size=0.025)

    train_name = outputfilename #"preprocess_trainval.txt"
    # test_name = "preprocess_test.txt"

    with ProcessPoolExecutor(max_workers=7) as executor:
        with open(os.path.join(output_dir, train_name), 'w') as fs:
            print("Preprocessing train and validation files")
            futures = {executor.submit(process_file, file, data_dir): file for file in trainval_files}
            for future in tqdm(as_completed(futures), total=len(trainval_files)):
                file = futures[future]
                try:
                    line = future.result()
                    fs.write(line + '\n')
                except Exception as exc:
                    print(f"{file} generated an exception: {exc}")






# import glob

# import cv2
# import os
# from tqdm import tqdm

# def extract_lip_embeddings(data_dir, input_file):
# 	import face_recognition
# 	video_path = data_dir + '/' + input_file
# 	identifier = video_path.split('/')[-1].split('.')[0]  # This will be the video identifier without file extension
# 	video_capture = cv2.VideoCapture(video_path)

# 	all_frame_data = ""

# 	frame_number = 0
# 	while video_capture.isOpened():
# 		ret, frame = video_capture.read()
# 		if not ret:
# 			break

# 		rgb_frame = frame[:, :, ::-1]
# 		face_landmarks_list = face_recognition.face_landmarks(rgb_frame, model="small")

# 		for face_landmarks in face_landmarks_list:
# 			if 'top_lip' in face_landmarks and 'bottom_lip' in face_landmarks:
# 				top_lip = face_landmarks['top_lip']
# 				bottom_lip = face_landmarks['bottom_lip']
# 				all_points = top_lip + bottom_lip
# 				x_coordinates = [p[0] for p in all_points]
# 				y_coordinates = [p[1] for p in all_points]
# 				left, right, top, bottom = min(x_coordinates), max(x_coordinates), min(y_coordinates), max(y_coordinates)

# 				# Collect coordinates for each frame
# 				all_frame_data += f"{left}/{right}/{top}/{bottom}/"

# 		frame_number += 1

# 	video_capture.release()
# 	return f"{input_file}/{all_frame_data}"

# from sklearn.model_selection import train_test_split
# def preprocess_video(data_dir, data_ext, output_dir, outputfilename):
# 	files = glob.glob(f'{data_dir}/*.{data_ext}', recursive= True)
# 	print(files)
# 	files = [os.path.relpath(file, data_dir) for file in files]

# 	trainval_files = files

# 	with open(output_dir + outputfilename, 'w') as fs:
# 		print("Preprocessing train and validation files")
# 		for file in tqdm(trainval_files):
# 			print(f"PROCESSING file {file}")
# 			lip_embedding_str = extract_lip_embeddings(data_dir, file)
# 			line = f"{lip_embedding_str}\n"
# 			fs.write(line)