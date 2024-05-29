
import glob
import face_recognition
import cv2
import os
from tqdm import tqdm

def extract_lip_embeddings(data_dir, input_file):
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

		frame_number += 1

	video_capture.release()
	return f"{input_file}/{all_frame_data}"

from sklearn.model_selection import train_test_split
def preprocess_video(data_dir, data_ext, output_dir, output_file):
	# 
	files = glob.glob(f'{data_dir}/*.{data_ext}', recursive= True)
	files = [os.path.relpath(file, data_dir) for file in files]

	trainval_files, test_files = train_test_split(files, test_size = 0.1)
	# train_files, val_files = train_test_split(trainval_files, test_size = 0.1)
	# last_dot_index = output_file.rfind('.')
	# base_name = output_file[:last_dot_index] if last_dot_index != -1 else output_file
	# extension = output_file[last_dot_index + 1:] if last_dot_index != -1 else ''

	train_name = "preprocess_trainval.txt"
	test_name = "preprocess_test.txt"

	with open(output_dir + train_name, 'w') as fs:
		print("Preprocessing train and validation files")
		for file in tqdm(trainval_files):
			print(f"PROCESSING file {file}")
			lip_embedding_str = extract_lip_embeddings(data_dir, file)
			line = f"{lip_embedding_str}\n"
			fs.write(line)

	with open(output_dir + test_name, 'w') as fs:
		print("Preprocessing test files")
		for file in tqdm(test_files):
			print(f"PROCESSING file {file}")
			lip_embedding_str = extract_lip_embeddings(data_dir, file)
			line = f"{lip_embedding_str}\n"
			fs.write(line)


