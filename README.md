# HERON Harmonization

This is the official repository for the HERON harmonization model.

## Data Preprocessing

1. Organize all `.nii` files under a single parent directory, with a subdirectory named after each site.

   ```text
   data/
   ├── site_01/
   │   ├── subjectA_001.nii
   │   ├── subjectA_002.nii
   │   └── ...
   ├── site_02/
   │   ├── subjectB_001.nii
   │   ├── subjectB_002.nii
   │   └── ...
   ├── site_03/
   │   ├── subjectC_001.nii
   │   ├── subjectC_002.nii
   │   └── ...
   └── ...
   ```

2. Update the `Your config` sections in `/Prepocess/generate_nii_json.py`, `/Prepocess/data_reader.py`, and `/Prepocess/split_traveler_4ssl.py` to specify the input data directory and the output directory for preprocessed data. `DATA_SPLIT` determines whether the processed data are used for training or testing. If your subject names contain site names, modify `split_traveler_4ssl.py` to remove site markers from the `subject` value during processing.
3. Run these three scripts in the order listed above to generate the preprocessed data, `preprocessed_datalist.json`, and `max_min_value.json`.

## Training HERON

1. Update `max_graylevel` and `min_graylevel` in `/configs/data_configs.py` using the maximum and minimum values in `max_min_value.json`.
2. Update `data_path`, `save_dir`, `experiment_description`, and `run_description` in `BE_ssl.sh`, `ME_ssl.sh`, and `Decoder_ssl.sh`. Set `data_path` to the directory containing your `preprocessed_datalist.json`. Training logs will be saved in the corresponding directory under `save_dir`. Use `hparam_overrides` to override parameters defined in `/configs/hparams.py`.
3. Run `BE_ssl.sh`, `ME_ssl.sh`, and `Decoder_ssl.sh` in this order to train HERON.

## Performing Harmonization

Run:

```bash
python -B Infer.py \
    --checkpoint_path checkpoint.pt \
    --test_datalist_path Your_preprocessed_datalist_for_test.json \
    --train_datalist_path Your_preprocessed_datalist_for_train.json \
    --center_site your_ref_site \
    --ref_subject your_ref_subject \
    --out_dir your_dir
```
