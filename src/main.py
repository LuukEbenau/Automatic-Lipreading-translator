

def parse_args():
	parser = argparse.ArgumentParser()
	# Do our argument parsing code here
	args = parser.parse_args()
	return args

# Main code
def main(args):
	# TODO: make some baseline code
	from vocoder import vocode

	print(vocode())

if __name__ == "__main__":
    args = parse_args()
    main(args)


