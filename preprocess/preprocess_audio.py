import os
import ffmpeg
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import shutil
import glob


def extract_audio_file(input_dir, output_dir, filename, from_ext, to_ext, samplerate):
    full_file_path = os.path.join(input_dir, filename)
    output_file_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.' + to_ext)

    try:
        (
            ffmpeg
            .input(full_file_path)
            .output(output_file_path, acodec='pcm_s16le', ac=1, ar=str(samplerate))  # Set audio codec, channel, and sample rate
            .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
        )
        return f"Extracted audio to {output_file_path}"
    except Exception as e:
        return f"Failed to extract audio from {filename}. Error: {e}"

def extract_audio(input_dir, output_dir, video_ext, audio_ext, samplerate):
    """This function is not used right now, since we got the audio files predelivered. but in case we want to change samlingrate, etc. this might be useful!"""

    from_ext = video_ext
    to_ext = audio_ext
    samplerate = samplerate
    # Ensure the target directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Collect the files to process
    files_to_process = [f for f in os.listdir(input_dir) if f.endswith(from_ext)]

    # Use ThreadPoolExecutor to process files concurrently
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(extract_audio_file, input_dir, output_dir, filename, from_ext, to_ext, samplerate): filename for filename in files_to_process}
        
        for future in as_completed(futures):
            filename = futures[future]
            try:
                result = future.result()
                print(result)
            except Exception as e:
                print(f"Failed to extract audio from {filename}. Error: {e}")




# def extract_audio(args):
# 	"""This function is not used right now, since we got the audio files predelivered. but in case we want to change samlingrate, etc. this might be useful!
# 	"""

# 	assert args.input_dir != None, 'Please supply input_dir, e.g. /mnt/d/Projects/kth/speechrecognition/project/Automatic-Lipreading-translator/data/GRID/pretrain/'
	
# 	assert args.output_dir != None, 'Please supply output_dir, e.g. /mnt/d/Projects/kth/speechrecognition/project/Automatic-Lipreading-translator/data/GRID_audio/pretrain/'

# 	from_ext = args.video_ext
# 	to_ext = args.audio_ext 
# 	samplerate = args.samplerate
# 	# Ensure the target directory exists
# 	os.makedirs(args.output_dir, exist_ok=True)
# 	import ffmpeg
# 	# Loop through all files in the from_dir
# 	for filename in os.listdir(args.input_dir):
# 		if filename.endswith(from_ext):
# 			full_file_path = os.path.join(args.input_dir, filename)
# 			output_file_path = os.path.join(args.output_dir, os.path.splitext(filename)[0] + '.' + to_ext)
			
# 			try:
# 				(
# 						ffmpeg
# 						.input(full_file_path)
# 						.output(output_file_path, acodec='pcm_s16le', ac=1, ar=str(samplerate))  # Set audio codec, channel, and sample rate
# 						.run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
# 				)
# 				print(f"Extracted audio to {output_file_path}")
# 			except Exception as e:
# 				print(f"Failed to extract audio from {filename}. Error: {e}")