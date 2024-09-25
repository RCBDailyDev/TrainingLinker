python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install .\WHL\1\torch-2.4.0+cu124-cp310-cp310-win_amd64.whl
pip install .\WHL\1\torchvision-0.19.0+cu124-cp310-cp310-win_amd64.whl
pip install .\WHL\1\xformers-0.0.28.dev890-cp310-cp310-win_amd64.whl
pip install --upgrade --reinstall .\WHL\1\bitsandbytes-0.43.3-py3-none-win_amd64.whl

pip install  --upgrade -r requirements.txt
#.\DSTpython\python.exe -m pip install  --force-reinstall tensorboard
pause