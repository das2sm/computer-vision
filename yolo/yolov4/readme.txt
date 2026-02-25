YOLOv4 Hardware & Environment Specifications
• Driver Version: 590

• CUDA Toolkit: 12.1

• cuDNN Version: 8.9.7

• Verification Status: Passed `./mnistCUDNN` test.

Workflow: Environment Activation

To work on the project, the virtual environment must be activated in every new terminal session:

• Conda: `conda activate yolo_env`

Python: Select Interpreter -> yolo_env

test if GPU detected:
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"}')"

Ensure PyTorch is installed specifically for CUDA 12.1 to maintain compatibility with the local hardware stack.
