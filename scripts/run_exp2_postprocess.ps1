cd C:\Users\Colonad\Desktop\CS547\CS_547_sam2_colonad
conda activate sam2gpu

$SAV  = "C:\Users\Colonad\Desktop\CS547\sav_val"
$VID  = Join-Path $SAV "JPEGImages_24fps"
$GT   = Join-Path $SAV "Annotations_6fps"

$BASELINE = (Resolve-Path "outputs\exp1_sav_val_pred").Path
$OUT2_REL = "outputs\exp2_sav_val_postprocessed"
New-Item -ItemType Directory -Force -Path $OUT2_REL | Out-Null
$OUT2 = (Resolve-Path $OUT2_REL).Path

# Optional cleanup of Windows-specific artifacts
Get-ChildItem -Recurse -Force "$GT"  | Where-Object { $_.Name -like "*Zone.Identifier*" } | Remove-Item -Force -ErrorAction SilentlyContinue
Get-ChildItem -Recurse -Force "$VID" | Where-Object { $_.Name -like "*Zone.Identifier*" } | Remove-Item -Force -ErrorAction SilentlyContinue
Get-ChildItem -Recurse -Force "$BASELINE" | Where-Object { $_.Name -like "*Zone.Identifier*" } | Remove-Item -Force -ErrorAction SilentlyContinue

python .\scripts\postprocess_sav_masks.py `
  --input_root "$BASELINE" `
  --output_root "$OUT2" `
  --min_area 64 `
  --close_radius 2 `
  --temporal_window 3

python .\sav_dataset\sav_evaluator.py --gt_root "$GT" --pred_root "$OUT2" -n 1 *> outputs\exp2_sav_val_metrics.txt

python .\scripts\compare_metrics.py `
  --baseline "outputs\exp1_sav_val_metrics.txt" `
  --new "outputs\exp2_sav_val_metrics.txt"