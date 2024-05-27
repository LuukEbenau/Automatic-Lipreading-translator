import face_recognition
import cv2
import argparse

def parse_args():
	parser = argparse.ArgumentParser()
	# Include additional arguments if needed
	parser.add_argument('--output', type=str, default='output2.txt', help='Output file to write the coordinates')
	parser.add_argument('--data_dir', type=str, default='/mnt/d/Projects/kth/speechrecognition/project/Automatic-Lipreading-translator/datasets/grid/train/main/s1', help='Output file to write the coordinates')
	args = parser.parse_args()
	return args

def main(args):
		# Define the video file path and the output path for bounding boxes
		output_path = '/mnt/d/Projects/kth/speechrecognition/project/Automatic-Lipreading-translator/outputs/'
		# video_file_path = 'swwc7a.mpg' 
		# video_path = args.data_dir + '/' + video_file_path
		# identifier = video_path.split('/')[-1].split('.')[0]  # This will be the video identifier without file extension

		from preprocess import preprocess_video
		preprocess_video(args.data_dir, 'mpg', output_path, args.output)

if __name__ == "__main__":
		args = parse_args()
		main(args)
