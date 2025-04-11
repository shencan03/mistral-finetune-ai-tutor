@echo off
REM Set up Python virtual environment and install dependencies

echo Creating virtual environment...
python -m venv .venv

echo Activating virtual environment...
call .venv\Scripts\activate

echo Upgrading pip...
pip install --upgrade pip

echo Installing requirements...
pip install -r requirements.txt

echo ✅ Setup complete. You can now run your scripts using:
echo call .venv\Scripts\activate
