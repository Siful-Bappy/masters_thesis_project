"""
Step 1 — Train the detector:
conda activate venv_masters_thesis_project
python src/train_detector.py
# → saves models/yolo11n_cards.pt

Step 2 — Train the card classifier:
python src/train_classifier.py
# → saves models/mobilenet_classifier.pt

Step 3 — Run tracking on a video:
python src/track.py --video dataset/bacarat_video.mp4
# → saves data/detections.csv + data/tracks.csv

Step 4 — Generate demo overlay video:
python src/demo_overlay.py --video dataset/bacarat_video.mp4
# → saves dataset_report/demo_bacarat_video.mp4

# Step 3: start training all 6 CRF models
python src/train_all_crfs.py

If you want to train just one specific CRF (e.g. only CRF 43):
python src/train_classifier.py --crf 43

After all training finishes, run the evaluation:
python src/eval_crf.py

This will show you the accuracy vs CRF table and save the plot to 
dataset_report/accuracy_vs_crf.png



"""