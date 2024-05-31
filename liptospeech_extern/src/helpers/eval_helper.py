import editdistance
def wer(predict, truth):
	word_pairs = [(p[0].split(' '), p[1].split(' ')) for p in zip(predict, truth)]
	wer = [1.0 * editdistance.eval(p[0], p[1]) / len(p[1]) for p in word_pairs]
	return wer