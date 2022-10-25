import pandas as pd
import argparse

def main(args):
	train_cer = pd.read_csv(args.train_cer_tsv, sep='\t')
	dev_cer = pd.read_csv(args.dev_cer_tsv, sep='\t')
	test_cer = pd.read_csv(args.test_cer_tsv, sep='\t')

	cer = pd.concat([train_cer, dev_cer, test_cer], ignore_index=True)

	# write to files
	with open(args.uer_char_tsv,'w') as f:
		f.write(cer.to_csv(sep='\t', index=False))

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--train-cer-tsv", default="uer_char.train.tsv", type=str)
	parser.add_argument("--dev-cer-tsv", default="uer_char.dev.tsv", type=str) 
	parser.add_argument("--test-cer-tsv", default="uer_char.test.tsv", type=str)
	parser.add_argument("--uer-char-tsv", default="uer_char.tsv", type=str)
	args = parser.parse_args()

	main(args)