python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install .\WHL\torch-2.4.1+cu124-cp312-cp312-win_amd64.whl
pip install .\WHL\torchvision-0.19.1+cu124-cp312-cp312-win_amd64.whl
pip install .\WHL\xformers-0.0.28.post1-cp312-cp312-win_amd64.whl
pip install --upgrade --reinstall .\WHL\bitsandbytes-0.44.1-py3-none-win_amd64.whl

pip install  --upgrade -r requirements.txt
#.\DSTpython\python.exe -m pip install  --force-reinstall tensorboard
pause