@echo off
chcp 65001
CALL %USERPROFILE%\anaconda3\Scripts\activate.bat workspace
python app.py  --port=2233  --share
pause