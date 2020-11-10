import argparse
from urban_sound_tagging_baseline.metrics import evaluate, micro_averaged_auprc, macro_averaged_auprc
from urban_sound_tagging_baseline.logging_exp import experiment_logger

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("annotation_path", type=str)
	parser.add_argument("prediction_path", type=str)
	parser.add_argument("yaml_path", type=str)
	parser.add_argument("model_name", type=str)
	parser.add_argument("dataset", type=str)

	args = parser.parse_args()
	
	mode_train = 'fine'
	logger = experiment_logger()
	logger.add_params({'model': args.model_name, 'dataset': args.dataset, 'mode': mode_train})	
	for mode in ("fine", "coarse"):

		df_dict = evaluate(args.prediction_path,
						   args.annotation_path,
						   args.yaml_path,
						   mode, logger=logger)
		
		micro_auprc, eval_df = micro_averaged_auprc(df_dict, return_df=True)
		macro_auprc, class_auprc = macro_averaged_auprc(df_dict, return_classwise=True)

		# Get index of first threshold that is at least 0.5
		thresh_0pt5_idx = (eval_df['threshold'] >= 0.5).nonzero()[0][0]

		print("{} level evaluation:".format(mode.capitalize()))
		print("======================")
		print(" * Micro AUPRC:		   {}".format(micro_auprc))
		print(" * Micro F1-score (@0.5): {}".format(eval_df["F"][thresh_0pt5_idx]))
		print(" * Macro AUPRC:		   {}".format(macro_auprc))
		print(" * Coarse Tag AUPRC:")

		for coarse_id, auprc in class_auprc.items():
			print("	  - {}: {}".format(coarse_id, auprc))
			logger.log_metrics(f"AUPRC_{mode}_{coarse_id}", auprc)