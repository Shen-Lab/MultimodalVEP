#!/bin/bash
python inference.py  --embedding_list esm2 esm_if gearnet --test_fold 0 --ckpt_path ../ckpt/esm2_struc/fold0/mlp_best_fold0.pt --dms_csv /DATA/yxliu/MEP-SiPLM/CAGI7/LPL.csv --output_file CAGI_LPL_GEAR.csv
python inference.py  --embedding_list esm2 esm_if gearnet --test_fold 0 --ckpt_path ../ckpt/esm2_struc/fold0/mlp_best_fold0.pt --dms_csv /DATA/yxliu/MEP-SiPLM/CAGI7/TSC2.csv --output_file CAGI_TSC2_GEAR.csv
# python inference.py  --embedding_list saprot --test_fold 0 --ckpt_path ../ckpt/saprot/fold0/mlp_best_fold0.pt --dms_csv /DATA/yxliu/MEP-SiPLM/CAGI7/LPL.csv --output_file CAGI_LPL_SA.csv
# python inference.py  --embedding_list saprot --test_fold 0 --ckpt_path ../ckpt/saprot/fold0/mlp_best_fold0.pt --dms_csv /DATA/yxliu/MEP-SiPLM/CAGI7/TSC2.csv --output_file CAGI_TSC2_SA.csv
# python inference.py  --embedding_list saprot esm_if gearnet --test_fold 0 --ckpt_path ../ckpt/saprot_struc/fold0/mlp_best_fold0.pt --dms_csv /DATA/yxliu/MEP-SiPLM/CAGI7/LPL.csv --output_file CAGI_LPL_SA_GEAR.csv
# python inference.py  --embedding_list saprot esm_if gearnet --test_fold 0 --ckpt_path ../ckpt/saprot_struc/fold0/mlp_best_fold0.pt --dms_csv /DATA/yxliu/MEP-SiPLM/CAGI7/TSC2.csv --output_file CAGI_TSC2_SA_GEAR.csv

