from src.data.vid_aud_lrs2 import MultiDataset as LRS2_Dataset
from src.data.vid_aud_lrs3 import MultiDataset as LRS3_Dataset
from src.data.vid_aud_grid import MultiDataset as GRID_Dataset

def get_dataset(args):
	if args.data_name == 'GRID':
		print("Selected GRID Dataset")
		train_data = GRID_Dataset(
			data=args.data,
			mode=args.mode,
			min_window_size=args.min_window_size,
			max_window_size=args.max_window_size,
			max_v_timesteps=args.max_timesteps,
			augmentations=args.augmentations
		)
	elif args.data_name == 'LRS2':
		train_data = LRS2_Dataset(
			data=args.data,
			mode=args.mode,
			min_window_size=args.min_window_size,
			max_window_size=args.max_window_size,
			max_v_timesteps=args.max_timesteps,
			augmentations=args.augmentations,
		)
	elif args.data_name == 'LRS3':
		train_data = LRS3_Dataset(
			data=args.data,
			mode=args.mode,
			min_window_size=args.min_window_size,
			max_window_size=args.max_window_size,
			max_v_timesteps=args.max_timesteps,
			augmentations=args.augmentations,
		)
	else:
		print(f"WARNING: Data name {args.data_name} not recognized")
		train_data = None
		
	return train_data