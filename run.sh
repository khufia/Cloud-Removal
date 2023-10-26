# # Train the model
# python train.py --nEpochs 10 --noise_type pernil

# # Inference
# python test.py --noise_type pernil


# # Train the model
# python train.py --nEpochs 10 --noise_type cloud

# # Inference
# python test.py --noise_type cloud

# Train the model
python train.py --nEpochs 10 --noise_type gaussian

# Inference
python test.py --noise_type gaussian

# Train the model
python train.py --nEpochs 50 --noise_type dec_intensity

# Inference
python test.py --noise_type dec_intensity