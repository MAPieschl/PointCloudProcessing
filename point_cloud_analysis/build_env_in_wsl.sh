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

    cd $PROJ_PATH

