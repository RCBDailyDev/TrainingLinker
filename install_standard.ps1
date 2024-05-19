python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
#pip install .\WHL\torch-2.3.0+cu118-cp310-cp310-win_amd64.whl
#pip install .\WHL\torchvision-0.18.0+cu118-cp310-cp310-win_amd64.whl
#pip install .\WHL\xformers-0.0.26.post1+cu118-cp310-cp310-win_amd64.whl
#pip install --upgrade --reinstall .\WHL\bitsandbytes-0.41.1-py3-none-win_amd64.whl

pip install  --upgrade -r requirements.txt
#.\DSTpython\python.exe -m pip install  --force-reinstall tensorboard
pause