echo "=== Installing FastAPI Packages ==="

if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "Virtual environment is active: $VIRTUAL_ENV"
else
    echo "Virtual environment not active. Activating..."
    source .venv/bin/activate
fi

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing FastAPI..."
pip install fastapi

echo "Installing Uvicorn..."
pip install uvicorn

echo "Installing SQLAlchemy..."
pip install sqlalchemy

echo "Installing Pydantic..."
pip install pydantic

echo -e "\n=== Verifying Installations ==="
pip list | grep -E "(fastapi|uvicorn|sqlalchemy|pydantic)"

echo -e "\n=== Testing Imports ==="
python -c "import fastapi; print('FastAPI imported successfully')"
python -c "import uvicorn; print('Uvicorn imported successfully')"
python -c "import sqlalchemy; print('SQLAlchemy imported successfully')"
python -c "import pydantic; print('Pydantic imported successfully')"

echo -e "\n All packages installed successfully!"
