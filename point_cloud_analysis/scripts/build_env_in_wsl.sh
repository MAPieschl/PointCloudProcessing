# Run this script in WSL to establish the environment for point_cloud_analysis

REQUIRED_PYTHON_VERSION="3.10.18"

PROJ_PATH=$(pwd)
echo "Project path is $package_path"

INSTALLED_PYTHON_DIST=$(python --version 2>&1)
INSTALLED_PYTHON_VERSION=$(echo "$INSTALLED_PYTHON_DIST" | awk '{print $2}')

echo "Installed Python version is $INSTALLED_PYTHON_VERSION"

if [ "$INSTALLED_PYTHON_VERSION" != "$REQUIRED_PYTHON_VERSION" ]; then
    echo "Installing Python $REQUIRED_PYTHON_VERSION"

    cd /mnt/c

    # Install Python 3.10.18
    sudo apt update
    sudo apt install build-essential libssl-dev zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libreadline-dev libffi-dev wget

    wget https://www.python.org/ftp/python/3.10.18/Python-3.10.18.tgz
    tar -xf Python-3.10.18.tgz
    cd Python-3.10.14

    ./configure --enable-optimizations
    make -j$(nproc)
    sudo make altinstall

    NEW_VER=$(python3.10 --version 2>&1)

    echo "Installed Python $NEW_VER - use command 'python3.10' for REPL access"

    cd $PROJ_PATH

if [ $(pwd) == $PROJ_PATH ]; then
    echo "Initializing virtual environment..."
    python3.10 -m venv .venv

    echo "Entering virtual environment..."
    source .venv/bin/activate

    echo "Installing dependencies..."

    # Because of a conflicting dependency (protobuf) between
    # tensorflow and tf2onnx, tf2onnx MUST be installed first,
    # followed by tensorflow[and-cuda]
    python3.10 -m pip install --upgrade pip
    pip install -r requirements_no_tf2onnx_tensorflow.txt
    pip install tf2onnx
    pip install tensorflow