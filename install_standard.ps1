python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install .\WHL\torch-2.7.0+cu126-cp312-cp312-win_amd64.whl
pip install .\WHL\torchvision-0.22.0+cu126-cp312-cp312-win_amd64.whl
pip install .\WHL\xformers-0.0.30-cp312-cp312-win_amd64.whl
pip install --upgrade --reinstall .\WHL\bitsandbytes-0.45.0-py3-none-win_amd64.whl

pip install  --upgrade -r requirements.txt
#.\DSTpython\python.exe -m pip install  --force-reinstall tensorboard
pause